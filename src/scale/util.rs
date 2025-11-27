use num_traits::Float;

/// Return `(min, max)` for two owned values.
pub fn sorted_pair<T: PartialOrd>(a: T, b: T) -> (T, T) {
    if a <= b { (a, b) } else { (b, a) }
}

/// Return references ordered as `(min, max)` without cloning.
pub fn sorted_pair_refs<'a, T: PartialOrd>(a: &'a T, b: &'a T) -> (&'a T, &'a T) {
    if a <= b { (a, b) } else { (b, a) }
}

/// Compute a small epsilon relative to the provided step.
/// Returns step / 10, which is used as a tolerance for floating-point comparisons.
pub fn epsilon_from_step<T: Float>(step: &T) -> T {
    let ten = T::from(10.0).unwrap();
    *step / ten
}
