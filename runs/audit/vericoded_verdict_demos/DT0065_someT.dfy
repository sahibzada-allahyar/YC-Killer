function someT<T>(): T
type Empty = x: int | false witness *
lemma ConjureFalse() ensures false { var e: Empty := someT<Empty>(); }
