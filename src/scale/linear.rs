use super::{Scale, Tick, TickIter, util};
use num_traits::Float;

type TickGenerator<D, N> = Box<dyn Fn(&Linear<D, N>) -> TickIter<D>>;

/// Linear scale: affine mapping between a numeric domain and normalized `[0, 1]` range.
///
/// `Linear` provides a straightforward linear mapping between data values and
/// a normalized [0, 1] range. This is the most common scale type for charting.
///
/// # Type Parameters
///
/// - `D`: Domain type (the data values, typically `f32` or `f64`)
/// - `N`: Normalized type (typically `f32` or `f64`, represents `[0, 1]` range)
///
/// # Features
///
/// - **Bidirectional mapping**: Convert between domain and normalized values
/// - **Pan and zoom**: Interactively adjust the visible domain
/// - **Tick generation**: Automatically generate "nice" tick marks for axes
/// - **Reversed axes**: Support both increasing and decreasing domains
/// - **No clamping**: Out-of-range values map beyond [0, 1]
///
/// # Domain Ordering
///
/// Domain values are kept exactly as set (no implicit sorting), so both
/// normal and reversed axes are supported:
/// - Normal: `new(0.0, 100.0)` - larger values at the right/top
/// - Reversed: `new(100.0, 0.0)` - larger values at the left/bottom
///
/// # Examples
///
/// ## Basic Usage
///
/// ```rust
/// use aksel::{Scale, scale::Linear};
///
/// let scale = Linear::<f64, f64>::new(0.0, 100.0);
///
/// // Normalize to [0, 1]
/// assert_eq!(scale.normalize(&0.0), 0.0);
/// assert_eq!(scale.normalize(&50.0), 0.5);
/// assert_eq!(scale.normalize(&100.0), 1.0);
///
/// // Denormalize back to domain
/// assert_eq!(scale.denormalize(0.0), 0.0);
/// assert_eq!(scale.denormalize(0.5), 50.0);
/// assert_eq!(scale.denormalize(1.0), 100.0);
/// ```
///
/// ## Mixed Type Precision
///
/// ```rust
/// use aksel::{Scale, scale::Linear};
///
/// // f64 domain, f32 normalized (common for GPU rendering)
/// let scale = Linear::<f64, f32>::new(0.0, 100.0);
///
/// let normalized: f32 = scale.normalize(&50.0);
/// assert_eq!(normalized, 0.5f32);
/// ```
///
/// ## Pan and Zoom
///
/// ```rust
/// use aksel::{Scale, scale::Linear};
///
/// let mut scale = Linear::<f64, f64>::new(0.0, 100.0);
///
/// // Pan by 20% of the range
/// scale.pan(0.2);
/// assert_eq!(scale.domain(), (&20.0, &120.0));
///
/// // Zoom in 2x around the center
/// let mut scale = Linear::<f64, f64>::new(0.0, 100.0);
/// scale.zoom(2.0, Some(0.5));
/// assert_eq!(scale.domain(), (&25.0, &75.0));
///
/// // Zoom in 2x around the left edge (25% position)
/// let mut scale = Linear::<f64, f64>::new(0.0, 100.0);
/// scale.zoom(2.0, Some(0.25));
/// assert_eq!(scale.domain(), (&12.5, &62.5));
/// ```
///
/// ## Reversed Axis
///
/// ```rust
/// use aksel::{Scale, scale::Linear};
///
/// // Create a reversed scale (useful for Y-axes that go down)
/// let scale = Linear::<f64, f64>::new(100.0, 0.0);
///
/// assert_eq!(scale.normalize(&100.0), 0.0);
/// assert_eq!(scale.normalize(&0.0), 1.0);
/// ```
///
/// ## Custom Tick Generation
///
/// ```rust
/// use aksel::{Scale, scale::{Linear, Tick}};
///
/// // Create a scale with custom ticks
/// let scale = Linear::<f64, f64>::new_with_tick_fn(0.0, 100.0, |_scale| {
///     vec![
///         Tick { value: 0.0, level: 0 },
///         Tick { value: 50.0, level: 0 },
///         Tick { value: 100.0, level: 0 },
///     ]
/// });
///
/// assert_eq!(scale.ticks().len(), 3);
/// ```
///
/// ## Out-of-Range Values
///
/// ```rust
/// use aksel::{Scale, scale::Linear};
///
/// let scale = Linear::<f64, f64>::new(0.0, 100.0);
///
/// // Values outside domain are not clamped
/// assert_eq!(scale.normalize(&150.0), 1.5);
/// assert_eq!(scale.normalize(&-50.0), -0.5);
/// ```
pub struct Linear<D, N = f64>
where
    D: Float,
    N: Float,
{
    min: D,
    max: D,
    tick_generator: TickGenerator<D, N>,
    _phantom: std::marker::PhantomData<N>,
}

/// Helper to find a "nice" step size using a simple iterative approach.
/// Works directly with the generic type D without needing logarithms.
fn nice_step<D: Float>(raw_step: D) -> D {
    // Nice values to test: 1, 2, 5, 10, 20, 50, 100, etc.
    // We'll build these by repeatedly multiplying by 10, 5, 2
    let one = D::one();
    let two = one + one;
    let five = two + two + one;
    let ten = five + five;

    let abs_step = raw_step.abs();

    // Find a nice value close to raw_step
    // Start at 1 and scale up/down to find the right magnitude
    let mut candidate = one;

    // Scale up if raw_step is larger than our candidate
    while candidate * ten < abs_step {
        candidate = candidate * ten;
    }

    // Scale down if raw_step is smaller than our candidate / 10
    while candidate > abs_step * ten {
        candidate = candidate / ten;
    }

    // Now candidate is within an order of magnitude of abs_step
    // Try nice multiples: 1x, 2x, 5x, 10x of the base
    let candidates = [
        candidate,
        candidate * two,
        candidate * five,
        candidate * ten,
    ];

    // Pick the smallest candidate that is >= abs_step
    for c in candidates {
        if c >= abs_step {
            return c;
        }
    }

    // Fallback: return the largest candidate
    candidate * ten
}

const MAX_MINOR_TICKS: usize = 100_000;

pub struct LinearTickIter<D: Float> {
    state: LinearTickState<D>,
    remaining: usize,
}

enum LinearTickState<D: Float> {
    Single(Option<D>),
    Sweep(LinearSweepState<D>),
    Done,
}

struct LinearSweepState<D: Float> {
    start: D,
    end_tol: D,
    minor_step: D,
    current_index: usize,
    clamp_min: D,
    clamp_max: D,
    epsilon: D,
    last_value: Option<D>,
}

impl<D: Float> LinearTickIter<D> {
    pub(crate) fn from_scale<N: Float>(scale: &Linear<D, N>) -> Self {
        Self::new(scale.min, scale.max)
    }

    pub(crate) fn new(min: D, max: D) -> Self {
        if min == max {
            return Self {
                state: LinearTickState::Single(Some(min)),
                remaining: 1,
            };
        }

        let (mut lo, mut hi) = util::sorted_pair(min, max);
        let clamp_min = lo;
        let clamp_max = hi;

        let range = hi - lo;
        let rough_step = range / D::from(10.0).unwrap();
        let major_step = nice_step(rough_step);

        let lo_ratio = lo / major_step;
        lo = lo_ratio.floor() * major_step;

        let hi_ratio = hi / major_step;
        hi = hi_ratio.ceil() * major_step;

        let minor_step = major_step / D::from(10.0).unwrap();
        if minor_step == D::zero() {
            return Self {
                state: LinearTickState::Single(Some(lo)),
                remaining: 1,
            };
        }
        let epsilon = util::epsilon_from_step(&major_step);
        let end_tol = hi + epsilon;

        Self {
            state: LinearTickState::Sweep(LinearSweepState {
                start: lo,
                end_tol,
                minor_step,
                current_index: 0,
                clamp_min,
                clamp_max,
                epsilon,
                last_value: None,
            }),
            remaining: MAX_MINOR_TICKS,
        }
    }
}

impl<D: Float> Iterator for LinearTickIter<D> {
    type Item = Tick<D>;

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.state {
            LinearTickState::Single(slot) => slot.take().map(|value| Tick { value, level: 0 }),
            LinearTickState::Sweep(state) => {
                while self.remaining > 0 {
                    // Calculate value based on index to avoid accumulation errors
                    let mut value =
                        state.start + state.minor_step * D::from(state.current_index).unwrap();

                    if value > state.end_tol {
                        self.state = LinearTickState::Done;
                        return None;
                    }

                    self.remaining -= 1;

                    // Determine level using the index before incrementing
                    let index = state.current_index;
                    state.current_index += 1;

                    if value < state.clamp_min {
                        let diff = state.clamp_min - value;
                        if diff <= state.epsilon {
                            value = state.clamp_min;
                        } else {
                            continue;
                        }
                    } else if value > state.clamp_max {
                        let diff = value - state.clamp_max;
                        if diff <= state.epsilon {
                            value = state.clamp_max;
                        } else {
                            self.state = LinearTickState::Done;
                            return None;
                        }
                    }

                    let level = if index % 10 == 0 { 0 } else { 1 };

                    if state.last_value.map(|last| last == value).unwrap_or(false) {
                        continue;
                    }
                    state.last_value = Some(value);

                    return Some(Tick { value, level });
                }

                self.state = LinearTickState::Done;
                None
            }
            LinearTickState::Done => None,
        }
    }
}

fn default_tick_generator<D: Float + 'static, N: Float>(scale: &Linear<D, N>) -> TickIter<D> {
    TickIter::from_linear(LinearTickIter::from_scale(scale))
}

impl<D, N> Linear<D, N>
where
    D: Float + 'static,
    N: Float + 'static,
{
    /// Creates a new linear scale with the given domain range.
    ///
    /// Uses the default tick generator which creates "nice" tick marks
    /// at round numbers (e.g., 0, 10, 20, 50, 100).
    ///
    /// # Arguments
    ///
    /// * `min` - The minimum value of the domain
    /// * `max` - The maximum value of the domain
    ///
    /// # Examples
    ///
    /// ```
    /// use aksel::{Scale, scale::Linear};
    ///
    /// let scale = Linear::<f64, f64>::new(0.0, 100.0);
    /// assert_eq!(scale.domain(), (&0.0, &100.0));
    /// ```
    pub fn new(min: D, max: D) -> Self {
        Self {
            min,
            max,
            tick_generator: Box::new(default_tick_generator),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Creates a new linear scale with a custom tick generator.
    ///
    /// The tick generator is a function that takes a reference to the scale
    /// and returns a [`TickIter`] that generates tick marks.
    ///
    /// # Arguments
    ///
    /// * `min` - The minimum value of the domain
    /// * `max` - The maximum value of the domain
    /// * `tick_generator` - A function that generates ticks for this scale
    ///
    /// # Examples
    ///
    /// ```
    /// use aksel::{Scale, scale::{Linear, TickIter}};
    ///
    /// let scale = Linear::<f64, f64>::new_with_tick_generator(0.0, 100.0, |_scale| {
    ///     TickIter::empty()
    /// });
    /// assert!(scale.ticks().is_empty());
    /// ```
    pub fn new_with_tick_generator<F>(min: D, max: D, tick_generator: F) -> Self
    where
        F: Fn(&Self) -> TickIter<D> + 'static,
    {
        Self {
            min,
            max,
            tick_generator: Box::new(tick_generator),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Creates a new linear scale with a custom tick function.
    ///
    /// This is a convenience method that wraps a function returning `Vec<Tick<D>>`
    /// into a tick generator. Use this when you want to provide a simple list
    /// of ticks rather than implementing an iterator.
    ///
    /// # Arguments
    ///
    /// * `min` - The minimum value of the domain
    /// * `max` - The maximum value of the domain
    /// * `tick_fn` - A function that generates a vector of ticks
    ///
    /// # Examples
    ///
    /// ```
    /// use aksel::{Scale, scale::{Linear, Tick}};
    ///
    /// let scale = Linear::<f64, f64>::new_with_tick_fn(0.0, 100.0, |_scale| {
    ///     vec![
    ///         Tick { value: 0.0, level: 0 },
    ///         Tick { value: 50.0, level: 0 },
    ///         Tick { value: 100.0, level: 0 },
    ///     ]
    /// });
    /// assert_eq!(scale.ticks().len(), 3);
    /// ```
    pub fn new_with_tick_fn<F>(min: D, max: D, tick_fn: F) -> Self
    where
        F: Fn(&Self) -> Vec<Tick<D>> + 'static,
    {
        Self::new_with_tick_generator(min, max, move |scale| TickIter::from_vec(tick_fn(scale)))
    }
}

impl<D, N> Scale for Linear<D, N>
where
    D: Float,
    N: Float,
{
    type Domain = D;
    type Normalized = N;

    fn domain(&self) -> (&D, &D) {
        (&self.min, &self.max)
    }

    fn set_domain(&mut self, min: D, max: D) {
        self.min = min;
        self.max = max;
    }

    fn normalize_opt(&self, value: &D) -> Option<N> {
        let span = self.max - self.min;
        if span == D::zero() {
            return Some(N::zero());
        }

        let offset = *value - self.min;
        // Convert offset and span to normalized type
        let offset_n: N = N::from(offset)?;
        let span_n: N = N::from(span)?;

        // NOTE: Intentionally no clamping here; we want out-of-range values
        // to map to <0 or >1 so the renderer can decide how to handle them.
        Some(offset_n / span_n)
    }

    fn denormalize_opt(&self, t: N) -> Option<D> {
        let span = self.max - self.min;
        let span_n: N = N::from(span)?;
        let scaled = t * span_n;
        let scaled_d: D = D::from(scaled)?;
        Some(self.min + scaled_d)
    }

    fn pan_opt(&mut self, delta_norm: N) -> Option<()> {
        // Shift the domain by delta_norm * span.
        let span = self.max - self.min;
        let span_n: N = N::from(span)?;
        let shift_n = span_n * delta_norm;
        let shift: D = D::from(shift_n)?;

        self.min = self.min + shift;
        self.max = self.max + shift;
        Some(())
    }

    fn zoom_opt(&mut self, factor: N, anchor_norm: Option<N>) -> Option<()> {
        // factor <= 0 is nonsensical; return None.
        if factor <= N::zero() {
            return None;
        }

        // Default anchor at center: 0.5
        let one = N::one();
        let two = one + one;
        let half = one / two;
        let anchor_norm = anchor_norm.unwrap_or(half);

        // Current span and anchor value in domain space.
        let span = self.max - self.min;
        let anchor_val = self.denormalize_opt(anchor_norm)?;

        // New span = old span / factor
        let span_n: N = N::from(span)?;
        let new_span_n = span_n / factor;

        // Split new span around anchor based on normalized anchor position.
        let left_frac_n = anchor_norm;
        let right_frac_n = one - anchor_norm;

        let left_shift: D = D::from(new_span_n * left_frac_n)?;
        let right_shift: D = D::from(new_span_n * right_frac_n)?;

        let new_min = anchor_val - left_shift;
        let new_max = anchor_val + right_shift;

        self.min = new_min;
        self.max = new_max;
        Some(())
    }

    fn tick_iter(&self) -> TickIter<D> {
        (self.tick_generator)(self)
    }

    fn extend_domain(&mut self, other_min: &D, other_max: &D) {
        if other_min < &self.min {
            self.min = *other_min;
        }
        if other_max > &self.max {
            self.max = *other_max;
        }
    }

    fn is_valid_domain_value(&self, _value: &D) -> bool {
        // Linear scale accepts any value in the numeric type.
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_normalize_f64() {
        let scale = Linear::<f64, f64>::new(0.0, 100.0);

        assert_eq!(scale.normalize(&0.0), 0.0);
        assert_eq!(scale.normalize(&50.0), 0.5);
        assert_eq!(scale.normalize(&100.0), 1.0);
        assert_eq!(scale.normalize(&25.0), 0.25);
        assert_eq!(scale.normalize(&75.0), 0.75);
    }

    #[test]
    fn test_linear_normalize_f32() {
        let scale = Linear::<f32, f32>::new(0.0, 100.0);

        assert_eq!(scale.normalize(&0.0), 0.0);
        assert_eq!(scale.normalize(&50.0), 0.5);
        assert_eq!(scale.normalize(&100.0), 1.0);
    }

    #[test]
    fn test_linear_denormalize() {
        let scale = Linear::<f64, f64>::new(0.0, 100.0);

        assert_eq!(scale.denormalize(0.0), 0.0);
        assert_eq!(scale.denormalize(0.5), 50.0);
        assert_eq!(scale.denormalize(1.0), 100.0);
    }

    #[test]
    fn test_linear_reversed() {
        // Reversed scale (max < min)
        let scale = Linear::<f64, f64>::new(100.0, 0.0);

        assert_eq!(scale.normalize(&100.0), 0.0);
        assert_eq!(scale.normalize(&50.0), 0.5);
        assert_eq!(scale.normalize(&0.0), 1.0);
    }

    #[test]
    fn test_linear_pan() {
        let mut scale = Linear::<f64, f64>::new(0.0, 100.0);

        // Pan by 10% (should shift by 10 units)
        scale.pan(0.1);

        let (min, max) = scale.domain();
        assert_eq!(*min, 10.0);
        assert_eq!(*max, 110.0);
    }

    #[test]
    fn test_linear_zoom_in() {
        let mut scale = Linear::<f64, f64>::new(0.0, 100.0);

        // Zoom in by 2x at center
        scale.zoom(2.0, Some(0.5));

        let (min, max) = scale.domain();
        assert_eq!(*min, 25.0);
        assert_eq!(*max, 75.0);
    }

    #[test]
    fn test_linear_zoom_out() {
        let mut scale = Linear::<f64, f64>::new(0.0, 100.0);

        // Zoom out by 0.5x (makes range 2x bigger)
        scale.zoom(0.5, Some(0.5));

        let (min, max) = scale.domain();
        assert_eq!(*min, -50.0);
        assert_eq!(*max, 150.0);
    }

    #[test]
    fn test_linear_ticks_basic() {
        let scale = Linear::<f64, f64>::new(0.0, 100.0);
        let ticks = scale.ticks();

        // Should have major and minor ticks
        assert!(!ticks.is_empty());

        // Check that we have major ticks (level 0)
        let majors: Vec<_> = ticks.iter().filter(|t| t.level == 0).collect();
        assert!(!majors.is_empty());

        // Check that major ticks include boundaries
        let major_values: Vec<_> = majors.iter().map(|t| t.value).collect();
        assert!(major_values.contains(&0.0));
        assert!(major_values.contains(&100.0));
        println!("Majors: {majors:#?}");
        println!(
            "Minors: {:#?}",
            ticks.iter().filter(|t| t.level == 1).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_linear_ticks_f32() {
        let scale = Linear::<f32, f32>::new(0.0, 100.0);
        let ticks = scale.ticks();

        assert!(!ticks.is_empty());

        // Verify ticks are sorted
        for i in 1..ticks.len() {
            assert!(ticks[i].value >= ticks[i - 1].value);
        }
    }

    #[test]
    fn test_linear_ticks_remain_within_domain() {
        let scale = Linear::<f64, f64>::new(13.2, 47.8);
        let (min, max) = scale.domain();

        for tick in scale.ticks() {
            assert!(
                tick.value >= *min && tick.value <= *max,
                "tick {} outside domain [{}, {}]",
                tick.value,
                min,
                max
            );
        }
    }

    #[test]
    fn test_linear_ticks_do_not_overlap_levels() {
        let scale = Linear::<f64, f64>::new(13.2, 47.8);
        let mut seen: Vec<(f64, u8)> = Vec::new();

        for tick in scale.ticks() {
            if let Some((_, prev_level)) = seen.iter().find(|(v, _)| *v == tick.value) {
                assert_eq!(
                    *prev_level, tick.level,
                    "tick value {} emitted at both level {} and {}",
                    tick.value, prev_level, tick.level
                );
            } else {
                seen.push((tick.value, tick.level));
            }
        }
    }

    #[test]
    fn test_linear_extend_domain() {
        let mut scale = Linear::<f64, f64>::new(10.0, 20.0);

        // Extend to include 0..30
        scale.extend_domain(&0.0, &30.0);

        let (min, max) = scale.domain();
        assert_eq!(*min, 0.0);
        assert_eq!(*max, 30.0);
    }

    #[test]
    fn test_linear_mixed_types() {
        // Domain is f64, Normalized is f32
        let scale = Linear::<f64, f32>::new(0.0, 100.0);

        let normalized: f32 = scale.normalize(&50.0);
        assert_eq!(normalized, 0.5f32);

        let denormalized: f64 = scale.denormalize(0.5f32);
        assert_eq!(denormalized, 50.0);
    }
}
