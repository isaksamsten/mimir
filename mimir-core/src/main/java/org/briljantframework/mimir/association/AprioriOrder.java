/**
 * The MIT License (MIT)
 *
 * Copyright (c) 2016 Isak Karlsson
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
 * associated documentation files (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge, publish, distribute,
 * sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or
 * substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
 * NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
package org.briljantframework.mimir.association;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

import org.apache.commons.math3.util.Precision;

import com.carrotsearch.hppc.IntObjectMap;
import com.carrotsearch.hppc.IntObjectOpenHashMap;

/**
 * Created by isak on 2017-04-21.
 */
public class AprioriOrder {

  private static class FinalRule {
    private final ItemSet x, y;
    private final double support, supportRatio, confidence, lift;

    public FinalRule(ItemSetRule rule) {
      this.x = rule.consequent;
      this.y = rule.antecedent;
      this.support = rule.support;
      this.supportRatio = rule.supportRatio;
      this.confidence = rule.getORconf();
      this.lift = rule.getLift();
    }

    public ItemSet getX() {
      return x;
    }

    public ItemSet getY() {
      return y;
    }

    public double getSupport() {
      return support;
    }

    public double getSupportRatio() {
      return supportRatio;
    }

    public double getConfidence() {
      return confidence;
    }

    public double getLift() {
      return lift;
    }
  }

  private static class ItemSetRule {

    private final BitSet transactions;
    private final ItemSet consequent, antecedent;

    private final double support, supportRatio, ORconf;
    private final double lift;

    public ItemSetRule(ItemSet consequent, ItemSet antecedent, BitSet transactions, double ruleSup,
        double supportRatio, double confidence, double lift) {
      this.consequent = consequent;
      this.antecedent = antecedent;
      this.transactions = transactions;
      this.ORconf = confidence;
      this.support = ruleSup;
      this.supportRatio = supportRatio;
      this.lift = lift;
    }

    public double getSupport() {
      return support;
    }

    public double getSupportRatio() {
      return supportRatio;
    }

    public double getORconf() {
      return ORconf;
    }

    public double getLift() {
      return lift;
    }

    public double frequency() {
      return transactions.cardinality();
    }

    @Override
    public String toString() {
      return "{<" + consequent + " => " + antecedent + "> freq: " + transactions + "}";
    }
  }

  private static class ItemPosition {
    private final Map<Integer, Position> itemPosition;

    public ItemPosition() {
      this.itemPosition = new HashMap<>();
    }

    public void putAndExpand(int item, Position position) {
      Position pos = itemPosition.get(item);
      if (pos == null) {
        itemPosition.put(item, position);
      } else {
        pos.expand(position);
      }
    }

    public Position get(int item) {
      return itemPosition.get(item);
    }

    @Override
    public String toString() {
      return "ItemPosition{" + "itemPosition=" + itemPosition + '}';
    }
  }

  private static class Position {
    private int first, last;

    public Position(int first, int last) {
      this.first = first;
      this.last = last;
    }

    public int getFirst() {
      return first;
    }

    public int getLast() {
      return last;
    }

    public void expand(Position other) {
      this.first = Math.min(this.first, other.first);
      this.last = Math.max(this.last, other.last);
    }

    @Override
    public boolean equals(Object o) {
      if (this == o)
        return true;
      if (o == null || getClass() != o.getClass())
        return false;

      Position that = (Position) o;
      if (first != that.first)
        return false;
      return last == that.last;
    }

    @Override
    public int hashCode() {
      int result = first;
      result = 31 * result + last;
      return result;
    }

    @Override
    public String toString() {
      return "{" + "first=" + first + ", last=" + last + '}';
    }
  }

  private static class ItemSet {

    private final int[] items;
    private double support;

    private ItemSet(int[] items) {
      this.items = items;
    }

    public int get(int i) {
      return items[i];
    }

    public int size() {
      return items.length;
    }

    public boolean prefixMatch(ItemSet other, int prefixSize) {
      // assume sorted and both have size < prefixSize
      for (int i = 0; i < prefixSize; i++) {
        if (items[i] != other.items[i]) {
          return false;
        }
      }
      return true;
    }

    public boolean contains(ItemSet other) {
      int i = 0, j = 0;
      while (i < other.items.length && j < this.items.length) {
        if (this.items[j] < other.items[i]) {
          j++;
        } else if (this.items[j] == other.items[i]) {
          j++;
          i++;
        } else if (this.items[j] > other.items[i]) {
          return false;
        }
      }
      return i >= other.items.length;
    }

    public ItemSet merge(ItemSet b, int prefixSize) {
      int[] union = new int[this.items.length + 1];
      System.arraycopy(this.items, 0, union, 0, prefixSize);
      if (this.items[prefixSize] < b.items[prefixSize]) {
        union[prefixSize] = this.items[prefixSize];
        union[prefixSize + 1] = b.items[prefixSize];
      } else {
        union[prefixSize] = b.items[prefixSize];
        union[prefixSize + 1] = this.items[prefixSize];
      }
      return new ItemSet(union);
    }

    @Override
    public boolean equals(Object o) {
      if (this == o)
        return true;
      if (o == null || getClass() != o.getClass())
        return false;

      ItemSet other = (ItemSet) o;
      return Arrays.equals(items, other.items);
    }

    @Override
    public int hashCode() {
      return Arrays.hashCode(items);
    }

    @Override
    public String toString() {
      return "{" + Arrays.stream(items).mapToObj(Integer::toString).collect(Collectors.joining(","))
          + "}";
    }
  }

  private static class OrderedItemSet {

    // private final List<ItemSetRule> rules;
    private ItemSet itemSet;
    private final BitSet transactions;

    public OrderedItemSet(ItemSet item, BitSet transactions) {
      this.itemSet = item;
      // this.rules = new ArrayList<>();
      this.transactions = transactions;
    }

    // private OrderedItemSet(ItemSet itemSet, BitSet transactions, List<ItemSetRule> rules) {
    // this.itemSet = itemSet;
    //// this.rules = rules;
    // this.transactions = transactions;
    // }

    public double frequency() {
      return transactions.cardinality();
    }

    public BitSet getTransactions() {
      return transactions;
    }

    public ItemSet getItemSet() {
      return itemSet;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o)
        return true;
      if (o == null || getClass() != o.getClass())
        return false;

      OrderedItemSet itemSet = (OrderedItemSet) o;
      return this.itemSet.equals(itemSet.itemSet);
    }

    @Override
    public int hashCode() {
      return itemSet.hashCode();
    }

    @Override
    public String toString() {
      return "#{" + itemSet + " freq:" + frequency() + "}";
    }
  }

  public static BitSet intersection(BitSet a, BitSet b) {
    BitSet clone;
    BitSet and;
    if (a.size() < b.size()) {
      clone = a;
      and = b;
    } else {
      clone = b;
      and = a;
    }

    BitSet intersection = (BitSet) clone.clone();
    intersection.and(and);
    return intersection;
  }

  public static void main(String[] args) throws IOException {
    String file = "/Users/isak/Desktop/test_seq2.txt";
    // file = "/Volumes/Untitled 1/val/events/test3.seq";
    // file = "/Users/isak/Desktop/sample_ed2.seq";
    // file = "/Users/isak/Desktop/professors.seq";
    // file = "/Users/isak/Downloads/kosarak_2.txt";
    file = "/Users/isak/Downloads/kosarak25k.txt";
    TransactionSet transactionSet = readTransactions(file, true);
    double minSup = 0.002;
    double minSupRatio = 0.0;
    double minConf = 0.4;
    int orderConstraint = 1;

    long start = System.currentTimeMillis();
    List<FinalRule> rules =
        findFrequent(transactionSet, minSup, minSupRatio, orderConstraint, minConf);

    PrintStream out = new PrintStream(new FileOutputStream("/Users/isak/Desktop/result.csv"));
    // PrintStream out = System.out;
    System.out.printf("Found %d rules in %d ms with %.4f MB memory %n", rules.size(),
        System.currentTimeMillis() - start,
        (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / 1024.0 / 1024.0);
    out.printf("Rule, support, supportRatio, conf, lift%n");
    rules.stream().sorted(Comparator.comparing(FinalRule::getSupport).reversed()).forEach(rule -> {
      out.printf("\"%s => %s\",%f,%f,%f,%f%n", rule.x, rule.y, rule.getSupport(),
          rule.getSupportRatio(), rule.getConfidence(), rule.getLift());
    });

    out.flush();
  }

  private static List<FinalRule> findFrequent(TransactionSet transactionSet, double minSup,
      double minSupRatio, int orderConstraint, double minConf) {
    Map<ItemSet, Double> supports = new HashMap<>();
    IntObjectMap<ItemPosition> itemPositionMap = transactionSet.itemPositions;
    double noTransactions = transactionSet.getTransactions();


    List<OrderedItemSet> currentLevel = new ArrayList<>();
    for (OrderedItemSet orderedItemSet : transactionSet.getItemSets()) {
      if (orderedItemSet.frequency() / noTransactions >= minSup) {
        currentLevel.add(orderedItemSet);
        supports.put(orderedItemSet.getItemSet(), orderedItemSet.frequency() / noTransactions);
      }
    }

    // List<List<OrderedItemSet>> levels = new ArrayList<>();
    // levels.add(currentLevel);

    List<FinalRule> finalRules = new ArrayList<>();
    List<OrderedItemSet> nextLevel = new ArrayList<>();

    Map<ItemSet, List<ItemSetRule>> currentLevelMap = new HashMap<>();
    Map<ItemSet, List<ItemSetRule>> prevLevelMap = new HashMap<>();

    int level = 1;
    while (true) {
      // System.err.println("processing level: " + (level - 1) + " with " + currentLevel.size()
      // + " items" + " and " + finalRules.size() + " so far");
      for (int i = 0; i < currentLevel.size(); i++) {
        List<ItemSetRule> matches = new ArrayList<>();
        for (int j = i + 1; j < currentLevel.size(); j++) {
          OrderedItemSet iItem = currentLevel.get(i);
          OrderedItemSet jItem = currentLevel.get(j);

          if (iItem.getItemSet().prefixMatch(jItem.getItemSet(), level - 1)) {
            BitSet intersectingTransactions =
                intersection(iItem.getTransactions(), jItem.getTransactions());
            if (minSup >= intersectingTransactions.cardinality() / noTransactions)
              continue;

            double itemsetSup = intersectingTransactions.cardinality() / noTransactions;
            ItemSet newItemSet = iItem.getItemSet().merge(jItem.getItemSet(), level - 1);
            supports.put(newItemSet, itemsetSup);

            matches.clear();
            List<ItemSetRule> iItemRules = prevLevelMap.get(iItem.getItemSet());
            if (iItemRules != null) {
              matches.addAll(iItemRules);
            }
            List<ItemSetRule> jItemRules = prevLevelMap.get(jItem.getItemSet());
            if (jItemRules != null) {
              matches.addAll(jItemRules);
            }
            List<ItemSetRule> rules = new ArrayList<>();
            if (level > 1) {
              int[] tmpItemSet = new int[newItemSet.size() - 1];
              for (int k = newItemSet.size() - 3; k >= 0; k--) {
                int tmpCnt = 0;
                for (int l = 0; l < newItemSet.size(); l++) {
                  if (k != l) {
                    tmpItemSet[tmpCnt++] = newItemSet.get(l);
                  }
                }

                List<ItemSetRule> itemSetRules = prevLevelMap.get(new ItemSet(tmpItemSet));
                if (itemSetRules != null) {
                  matches.addAll(itemSetRules);
                }
              }

              Set<ItemSet> usedConsequents = new HashSet<>();
              Set<ItemSet> usedAntecendents = new HashSet<>();
              for (int k = 0; k < matches.size(); k++) {
                List<ItemSetRule> consequents = new ArrayList<>();
                List<ItemSetRule> antecendents = new ArrayList<>();

                ItemSetRule r = matches.get(k);
                boolean checkedConsequent = usedConsequents.contains(r.consequent);
                boolean checkedAntecedent = usedAntecendents.contains(r.antecedent);


                if (!checkedConsequent) {
                  consequents.add(r);
                  usedConsequents.add(r.consequent);
                }

                if (!checkedAntecedent) {
                  antecendents.add(r);
                  usedAntecendents.add(r.antecedent);
                }

                for (int l = k + 1; l < matches.size(); l++) {
                  ItemSetRule o = matches.get(l);
                  if (!checkedConsequent && o.consequent.equals(r.consequent)) {
                    consequents.add(o);
                  }
                  if (!checkedAntecedent && o.antecedent.equals(r.antecedent)) {
                    antecendents.add(o);
                  }
                }

                // If the number of itemsets to merge for the antecedent
                if (consequents.size() == level + 1 - r.consequent.size()) {
                  BitSet inter = new BitSet();
                  inter.or(consequents.get(0).transactions);
                  for (int i1 = 1; i1 < consequents.size(); i1++) {
                    ItemSetRule rule = consequents.get(i1);
                    inter.and(rule.transactions);
                  }

                  double ruleSup = inter.cardinality() / noTransactions;
                  if (ruleSup >= minSup) {
                    double supportRatio =
                        inter.cardinality() / (double) intersectingTransactions.cardinality();
                    ItemSet mergedConsequent =
                        mergeConsequents(consequents, consequents.size() - 1);
                    if (usedAntecendents.contains(mergedConsequent)) {
                      continue;
                    }
                    double confidence = ruleSup / supports.get(r.consequent);
                    double lift = Double.NaN;
                    // ruleSup / (supports.get(mergedConsequent) * supports.get(r.consequent));
                    ItemSetRule rule = new ItemSetRule(r.consequent, mergedConsequent, inter,
                        ruleSup, supportRatio, confidence, lift);
                    rules.add(rule);
                    if (Precision.compareTo(confidence, minConf, 0.0001) >= 0
                        && supportRatio >= minSupRatio) {
                      finalRules.add(new FinalRule(rule));
                    }
                  }
                }
                if (antecendents.size() == level + 1 - r.antecedent.size()) {
                  BitSet inter = new BitSet();
                  inter.or(antecendents.get(0).transactions);
                  for (int i1 = 1; i1 < antecendents.size(); i1++) {
                    ItemSetRule rule = antecendents.get(i1);
                    inter.and(rule.transactions);
                  }

                  double ruleSup = inter.cardinality() / noTransactions;
                  if (ruleSup >= minSup) {
                    double supportRatio =
                        inter.cardinality() / (double) intersectingTransactions.cardinality();
                    ItemSet mergedAntecedents =
                        mergeAntecedent(antecendents, antecendents.size() - 1);
                    if (usedConsequents.contains(mergedAntecedents)) {
                      continue;
                    }
                    double ORconf = ruleSup / supports.get(mergedAntecedents);
                    double lift = Double.NaN;
                    // ruleSup / (supports.get(mergedAntecedents) * supports.get(r.antecedent));

                    ItemSetRule rule = new ItemSetRule(mergedAntecedents, r.antecedent, inter,
                        ruleSup, supportRatio, ORconf, lift);
                    rules.add(rule);
                    if (Precision.compareTo(rule.getORconf(), minConf, 0.0001) >= 0
                        && supportRatio >= minSupRatio) {
                      finalRules.add(new FinalRule(rule));
                    }

                  }
                }
              }
            } else {
              List<OrderedItemSet> matches2 = new ArrayList<>();
              matches2.add(iItem);
              matches2.add(jItem);
              for (int k = 0; k < matches2.size(); k++) {
                OrderedItemSet kOrder = matches2.get(k);
                OrderedItemSet mOrder = null;
                for (int m = 0; m < matches2.size(); m++) {
                  if (m == k) {
                    continue;
                  }
                  mOrder = matches2.get(m);
                }
                int itemA = kOrder.getItemSet().get(0);
                int itemB = mOrder.getItemSet().get(0);

                BitSet beforeIntersection = itemIntersect(orderConstraint, itemPositionMap,
                    intersectingTransactions, itemA, itemB);

                double ruleSup = beforeIntersection.cardinality() / noTransactions;
                // && Precision.compareTo(ORconf, minConf, 0.001) >= 0
                if (ruleSup >= minSup) {
                  double supportRatio = beforeIntersection.cardinality()
                      / (double) intersectingTransactions.cardinality();
                  double ORconf = ruleSup / supports.get(kOrder.getItemSet());
                  double lift = Double.NaN;
                  // ruleSup
                  // / (supports.get(kOrder.getItemSet()) * supports.get(mOrder.getItemSet()));
                  ItemSetRule rule = new ItemSetRule(kOrder.getItemSet(), mOrder.getItemSet(),
                      beforeIntersection, ruleSup, supportRatio, ORconf, lift);
                  rules.add(rule);
                  if (Precision.compareTo(rule.getORconf(), minConf, 0.0001) >= 0
                      && supportRatio >= minSupRatio) {
                    finalRules.add(new FinalRule(rule));
                  }
                }
              }
            }

            nextLevel.add(new OrderedItemSet(newItemSet, intersectingTransactions));
            currentLevelMap.put(newItemSet, rules);
          } else {
            break;
          }
        }
      }
      if (!nextLevel.isEmpty()) {
        // levels.add(nextLevel);
        currentLevel.clear();
        List<OrderedItemSet> tmp = currentLevel;
        currentLevel = nextLevel;
        nextLevel = tmp;

        prevLevelMap.clear();
        Map<ItemSet, List<ItemSetRule>> tmpMap = prevLevelMap;
        prevLevelMap = currentLevelMap;
        currentLevelMap = tmpMap;

        level += 1;
      } else {
        break;
      }
    }

    // for (int i = 1; i < levels.size(); i++) {
    // List<OrderedItemSet> item = levels.get(i);
    // for (OrderedItemSet orderedItemSet : item) {
    // for (ItemSetRule rule : orderedItemSet.rules) {
    // if (Precision.compareTo(rule.getORconf(), minConf, 0.0001) >= 0) {
    // finalRules.add(rule);
    // }
    // }
    // }
    // }

    return finalRules;
  }

  private static BitSet itemIntersect(int orderConstraint,
      IntObjectMap<ItemPosition> itemPositionMap, BitSet intersectingTransactions, int itemA,
      int itemB) {
    BitSet beforeIntersection = new BitSet();
    for (int transactionId =
        intersectingTransactions.nextSetBit(0); transactionId >= 0; transactionId =
            intersectingTransactions.nextSetBit(transactionId + 1)) {
      ItemPosition pos = itemPositionMap.get(transactionId);
      if (pos.get(itemB).getLast() - pos.get(itemA).getFirst() >= orderConstraint) {
        beforeIntersection.set(transactionId);
      }
      if (transactionId == Integer.MAX_VALUE) {
        break; // or (i+1) would overflow
      }
    }
    return beforeIntersection;
  }

  // private static void addRules(boolean before, double noTransactions, double minSup, int level,
  // Set<Integer> intersection, ItemSetRules beforeRules, ItemSet ant,
  // List<ItemSetRule> beforeConsequents) {
  // if (beforeConsequents.size() == level) {
  // Set<Integer> inter = new HashSet<>(intersection);
  // for (ItemSetRule antecedent : beforeConsequents) {
  // inter.retainAll(antecedent.transactions);
  // }
  //
  // if (inter.size() / noTransactions > minSup) {
  // ItemSet mergedConsequent = mergeConsequents(beforeConsequents, level - 1);
  // beforeRules.addRule(ant, mergedConsequent, before, inter);
  // }
  // }
  // }

  private static ItemSet mergeConsequents(List<ItemSetRule> consequents, int size) {
    int[] union = new int[size + 1];
    Arrays.fill(union, Integer.MAX_VALUE);
    for (int i = 0; i < size; i++) {
      for (ItemSetRule rule : consequents) {
        int cons = rule.antecedent.get(i);
        if (cons < union[i]) {
          union[i] = cons;
        }
        union[i + 1] = cons;
      }

    }
    return new ItemSet(union);
  }

  private static ItemSet mergeAntecedent(List<ItemSetRule> antecedents, int size) {
    int[] union = new int[size + 1];
    // Arrays.fill(union, Integer.MAX_VALUE);
    for (int i = 0; i < size; i++) {
      for (ItemSetRule rule : antecedents) {
        int cons = rule.consequent.get(i);
        if (cons < union[i] || union[i] == 0) {
          union[i] = cons;
        }
        union[i + 1] = cons;
      }

    }
    return new ItemSet(union);
  }


  private static class TransactionSet {
    private final List<OrderedItemSet> itemSets;
    private final double transactions;
    private final IntObjectMap<ItemPosition> itemPositions;


    private TransactionSet(List<OrderedItemSet> itemSets, double transactions,
        IntObjectMap<ItemPosition> itemPositions) {
      this.itemSets = itemSets;
      this.transactions = transactions;
      this.itemPositions = itemPositions;
    }

    public double getTransactions() {
      return transactions;
    }

    public List<OrderedItemSet> getItemSets() {
      return itemSets;
    }

    public ItemPosition getPosition(int transaction) {
      return itemPositions.get(transaction);
    }

  }


  private static int[][] transactions(String file, boolean seq) {
    Path path = Paths.get(file);
    try {
      List<String> lines = Files.readAllLines(path);
      int[][] transactions = new int[lines.size()][];
      for (int i = 0; i < lines.size(); i++) {
        if ("".equals(lines.get(i))) {
          transactions[i] = new int[0];
        } else {
          String[] line = lines.get(i).trim().split("\\s+");
          if (seq) {
            int size = 0;
            for (String aLine : line) {
              if ('-' != (aLine.trim().charAt(0))) {
                size++;
              }
            }
            transactions[i] = new int[size];
            int id = 0;
            for (String aLine : line) {
              if ('-' != (aLine.trim().charAt(0))) {
                transactions[i][id++] = Integer.parseInt(aLine.trim());
              }
            }
          } else {
            transactions[i] = new int[line.length];
            for (int j = 0; j < line.length; j++) {
              transactions[i][j] = Integer.parseInt(line[j].trim());
            }
          }
          // shuffleArray(transactions[i]);
        }
      }
      return transactions;
    } catch (IOException e) {
      e.printStackTrace();
    }
    return null;
  }

  private static List<List<Event>> readEventTransactions(String file) {
    Path path = Paths.get(file);
    try {
      List<List<Event>> transactions = new ArrayList<>();
      List<String> lines = Files.readAllLines(path);
      for (String line : lines) {
        List<Event> transaction = new ArrayList<>();
        String[] events = line.trim().split("\\s+");
        for (String event : events) {
          String[] split = event.split(",");
          transaction.add(new Event(Integer.parseInt(split[0]), Integer.parseInt(split[1])));
        }
        transactions.add(transaction);
      }
      return transactions;
    } catch (IOException e) {
      e.printStackTrace();
    }

    return null;
  }

  private static class Event {
    int value;
    int time;

    public Event(int value, int time) {
      this.value = value;
      this.time = time;
    }

    public int getValue() {
      return value;
    }

    public int getTime() {
      return time;
    }
  }

  private static TransactionSet readTransactions(String file, boolean b) {
    Map<Integer, BitSet> items = new HashMap<>();

    IntObjectMap<ItemPosition> itemPosition = new IntObjectOpenHashMap<>();

    // int[][] transactions = new int[][] {{1, 2, 3, 1, 1}, {2, 1, 3, 4, 3, 1, 5}, {1, 2, 3},
    // {3, 2, 1}, {2, 1, 3}, {5, 1, 3}, {2, 3, 5}, {1, 2, 6}, {2, 3, 4, 1, 2, 2, 2, 3, 4, 5}};

    // List<List<Event>> transactions = readEventTransactions("/Users/isak/Desktop/test_seq2.txt");
    List<List<Event>> transactions;
    if (b) {
      transactions = readSimpleEventTransactions(file);
    } else {
      transactions = readEventTransactions(file);
    }
    // int[][] transactions = new int[100][];
    // for (int i = 0; i < transactions.length; i++) {
    // transactions[i] = new int[ThreadLocalRandom.current().nextInt(10, 20)];
    // for (int j = 0; j < transactions[i].length; j++) {
    // transactions[i][j] = ThreadLocalRandom.current().nextInt(1, 20);
    // }
    // }
    // int[][] transactions = transactions("/Volumes/Untitled 1/val/events/test2.seq", false);
    // int[][] transactions = transactions("/Users/isak/Desktop/kosarak.dat.txt");
    // int[][] transactions = transactions("/Users/isak/Desktop/retail.txt");
    // int[][] transactions = transactions("/Users/isak/Desktop/accidents.txt", false);
    // int[][] transactions = transactions("/Users/isak/Desktop/SIGN.txt", true);
    // int[][] transactions = transactions("/Users/isak/Desktop/LEVIATHAN.txt", true);
    // int[][] transactions = transactions("/Users/isak/Desktop/foodmartFIM.txt");
    // int[][] transactions = transactions("/Users/isak/Downloads/OnlineRetail/OnlineRetail.txt",
    // false);
    // int ff = 0;
    // for (int[] transaction : transactions) {
    // System.out.println(ff++ + ": " + Arrays.toString(transaction));
    // }

    // System.out.println("Transaction count: " + transactions.length);
    // 1084> => <1097,1126,2183

    int transactionId = 0;
    for (List<Event> transaction : transactions) {
      ItemPosition pos = new ItemPosition();
      for (Event item : transaction) {
        if (items.containsKey(item.getValue())) {
          items.get(item.getValue()).set(transactionId);
        } else {
          // OrderedItemSet value = new OrderedItemSet(item);
          // value.inTransaction(transactionId);
          // Transactions trans = new Transactions();
          // trans.putAndExpand(transactionId, item, new Position(i, i));
          BitSet trans = new BitSet();
          trans.set(transactionId);
          items.put(item.getValue(), trans);
        }
        pos.putAndExpand(item.getValue(), new Position(item.getTime(), item.getTime()));
      }
      itemPosition.put(transactionId, pos);
      transactionId++;
    }

    // System.out.println("Unique items: " + items.size());

    // List<OrderedItemSet> items = new ArrayList<>();
    // items.add(new OrderedItemSet(2));
    // items.add(new OrderedItemSet(3));
    // items.add(new OrderedItemSet(1));
    ArrayList<OrderedItemSet> itemSets = new ArrayList<>();
    for (Map.Entry<Integer, BitSet> kv : items.entrySet()) {
      ItemSet itemSet = new ItemSet(new int[] {kv.getKey()});
      // itemSet.setSupport(kv.getValue().cardinality() / (double) transactions.length);
      itemSets.add(new OrderedItemSet(itemSet, kv.getValue()));
    }

    return new TransactionSet(itemSets, transactions.size(), itemPosition);
  }

  private static List<List<Event>> readSimpleEventTransactions(String file) {

    Path path = Paths.get(file);
    try {
      List<List<Event>> transactions = new ArrayList<>();
      List<String> lines = Files.readAllLines(path);
      for (String line : lines) {
        List<Event> transaction = new ArrayList<>();
        String[] events = line.trim().split("\\s+");
        for (int i = 0; i < events.length; i++) {
          String event = events[i];
          transaction.add(new Event(Integer.parseInt(event), i));
        }
        transactions.add(transaction);
      }
      return transactions;
    } catch (IOException e) {
      e.printStackTrace();
    }

    return null;
  }
}
