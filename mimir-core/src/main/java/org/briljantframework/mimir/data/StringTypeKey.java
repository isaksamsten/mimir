package org.briljantframework.mimir.data;

/**
 * Created by isak on 3/29/16.
 */
class StringTypeKey<T> implements TypeKey<T> {
  private final Class<T> cls;
  private final String name;
  private final T defaultValue;

  StringTypeKey(Class<T> cls, String name) {
    this(cls, name, null);
  }

  StringTypeKey(Class<T> cls, String name, T defaultValue) {
    this.cls = cls;
    this.name = name;
    this.defaultValue = defaultValue;
  }

  @Override
  public Class<T> getType() {
    return cls;
  }

  @Override
  public String getName() {
    return name;
  }

  @Override
  public String getDescription() {
    return name;
  }

  @Override
  public T defaultValue() {
    return defaultValue;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o)
      return true;
    if (o == null || getClass() != o.getClass())
      return false;

    StringTypeKey<?> that = (StringTypeKey<?>) o;
    if (cls != null ? !cls.equals(that.cls) : that.cls != null) {
      return false;
    } else {
      return name != null ? name.equals(that.name) : that.name == null;
    }
  }

  @Override
  public int hashCode() {
    int result = cls != null ? cls.hashCode() : 0;
    result = 31 * result + (name != null ? name.hashCode() : 0);
    return result;
  }
}
