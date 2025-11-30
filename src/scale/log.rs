use super::{Scale, Tick, TickIter, util};
use num_traits::Float;

type LogTickGenerator<D, N> = Box<dyn Fn(&Logarithmic<D, N>) -> TickIter<D>>;

/// Logarithmic scale: maps a numeric domain to normalized `[0, 1]` range using logarithmic transformation.
///
/// `Logarithmic` provides logarithmic mapping between data values and a normalized
/// [0, 1] range. This is essential for visualizing exponentially distributed data
/// such as scientific measurements, financial data, or population growth.
///
/// # Type Parameters
///
/// - `D`: Domain type (the data values, typically `f32` or `f64`)
/// - `N`: Normalized type (typically `f32` or `f64`, represents `[0, 1]` range)
///
/// # Features
///
/// - **Logarithmic mapping**: Equal distances in normalized space represent equal ratios in domain space
/// - **Base parameter**: Configurable logarithm base (typically 10 or e)
/// - **Pan and zoom**: Operations work in logarithmic space, preserving ratios
/// - **Tick generation**: Automatically generates ticks at powers of the base
/// - **Domain validation**: Only positive values are valid
///
/// # Domain Constraints
///
/// Logarithmic scales only accept **positive values** (> 0). Zero and negative
/// values are considered invalid:
/// - `is_valid_domain_value(&0.0)` returns `false`
/// - `normalize(&0.0)` returns `0.0` as a fallback
///
/// # Logarithmic Behavior
///
/// In logarithmic space:
/// - Equal distances represent equal **ratios** (multiplicative relationships)
/// - 1 → 10 has the same visual distance as 10 → 100
/// - Pan and zoom operations preserve geometric relationships
///
/// # Examples
///
/// ## Basic Usage
///
/// ```rust
/// use aksel::{Scale, scale::Logarithmic};
///
/// // Base-10 logarithmic scale from 1 to 1000
/// let scale = Logarithmic::<f64, f64>::new(10.0, 1.0, 1000.0);
///
/// // Values are distributed logarithmically
/// assert_eq!(scale.normalize(&1.0), 0.0);
/// assert_eq!(scale.normalize(&1000.0), 1.0);
///
/// // 10 is 1/3 of the way in log space (10^0 to 10^3)
/// let norm_10 = scale.normalize(&10.0);
/// assert!((norm_10 - 0.333).abs() < 0.01);
///
/// // 100 is 2/3 of the way in log space
/// let norm_100 = scale.normalize(&100.0);
/// assert!((norm_100 - 0.666).abs() < 0.01);
/// ```
///
/// ## Natural Logarithm (Base e)
///
/// ```rust
/// use aksel::{Scale, scale::Logarithmic};
///
/// let e = std::f64::consts::E;
/// let scale = Logarithmic::<f64, f64>::new(e, 1.0, e * e);
///
/// // ln(1) = 0, ln(e²) = 2
/// assert_eq!(scale.normalize(&1.0), 0.0);
/// assert!((scale.normalize(&e) - 0.5).abs() < 0.01);
/// ```
///
/// ## Pan and Zoom
///
/// Pan and zoom operations work in logarithmic space, which means they
/// preserve **ratios** rather than differences:
///
/// ```rust
/// use aksel::{Scale, scale::Logarithmic};
///
/// let mut scale = Logarithmic::<f64, f64>::new(10.0, 1.0, 100.0);
///
/// // Pan preserves the ratio (factor of 100)
/// scale.pan(0.1);
/// let (min, max) = scale.domain();
/// let ratio = max / min;
/// assert!((ratio - 100.0).abs() < 0.1);
///
/// // Zoom preserves geometric center
/// let mut scale = Logarithmic::<f64, f64>::new(10.0, 1.0, 100.0);
/// scale.zoom(2.0, Some(0.5));
/// let (min, max) = scale.domain();
/// let geometric_center = (min * max).sqrt();
/// assert!((geometric_center - 10.0).abs() < 0.1);
/// ```
///
/// ## Tick Generation
///
/// Logarithmic scales generate ticks at powers of the base:
///
/// ```rust
/// use aksel::{Scale, scale::Logarithmic};
///
/// let scale = Logarithmic::<f64, f64>::new(10.0, 1.0, 1000.0);
/// let ticks = scale.ticks();
///
/// // Major ticks at powers of 10: 1, 10, 100, 1000
/// let majors: Vec<_> = ticks.iter()
///     .filter(|t| t.level == 0)
///     .map(|t| t.value)
///     .collect();
///
/// assert!(majors.contains(&1.0));
/// assert!(majors.contains(&10.0));
/// assert!(majors.contains(&100.0));
/// assert!(majors.contains(&1000.0));
///
/// // Minor ticks at 2, 3, 4, ..., 20, 30, 40, ...
/// let minors: Vec<_> = ticks.iter()
///     .filter(|t| t.level == 1)
///     .collect();
/// assert!(!minors.is_empty());
/// ```
///
/// ## Domain Validation
///
/// ```rust
/// use aksel::{Scale, scale::Logarithmic};
///
/// let scale = Logarithmic::<f64, f64>::new(10.0, 1.0, 100.0);
///
/// // Only positive values are valid
/// assert!(scale.is_valid_domain_value(&1.0));
/// assert!(scale.is_valid_domain_value(&0.001));
/// assert!(!scale.is_valid_domain_value(&0.0));
/// assert!(!scale.is_valid_domain_value(&-10.0));
///
/// // Invalid values normalize to 0
/// assert_eq!(scale.normalize(&0.0), 0.0);
/// assert_eq!(scale.normalize(&-5.0), 0.0);
/// ```
///
/// ## Use Cases
///
/// Logarithmic scales are ideal for:
/// - Scientific data spanning multiple orders of magnitude
/// - Financial charts (stock prices, economic indicators)
/// - Population or growth data
/// - Audio levels (decibels)
/// - Seismic activity (Richter scale)
/// - pH scales in chemistry
///
/// ```rust
/// use aksel::{Scale, scale::Logarithmic};
///
/// // Example: Visualizing earthquake magnitudes (1 to 10 on Richter scale)
/// // Each unit represents a 10x increase in amplitude
/// let richter = Logarithmic::<f64, f64>::new(10.0, 1.0, 10.0);
///
/// // Magnitude 5 earthquake
/// let mag5 = richter.normalize(&5.0);
/// // Magnitude 6 earthquake (10x stronger)
/// let mag6 = richter.normalize(&6.0);
/// ```
pub struct Logarithmic<D, N = f64>
where
    D: Float,
    N: Float,
{
    base: D,
    min: D,
    max: D,
    tick_generator: LogTickGenerator<D, N>,
    _phantom: std::marker::PhantomData<N>,
}

fn log_ticks_exponent_range<D>(min: &D, max: &D, base: &D) -> Option<(i32, i32)>
where
    D: Float,
{
    let zero = D::zero();
    let one = D::one();
    let ten = D::from(10.0).unwrap();

    let (lo, hi) = util::sorted_pair_refs(min, max);

    // Validate inputs
    if lo <= &zero || hi <= &zero || base <= &zero {
        return None;
    }

    // Check if base is too close to 1 (use a small epsilon)
    let eps = one / ten;
    let eps_small = eps / (ten * ten);

    if (*base - one).abs() < eps_small {
        return None;
    }

    let ln_min = lo.ln();
    let ln_max = hi.ln();
    let ln_base = base.ln();

    let e_min_ratio = ln_min / ln_base;
    let e_max_ratio = ln_max / ln_base;

    let e_min = e_min_ratio.floor();
    let e_max = e_max_ratio.ceil();

    // Convert to i32 by successive addition/subtraction
    // This is a bit clunky but works without requiring a generic to_i32 method
    let mut e_min_i32 = 0i32;
    let mut counter = zero;

    // Count up from zero to e_min (or down if negative)
    if e_min >= zero {
        while counter < e_min && e_min_i32 < 1000 {
            counter = counter + one;
            e_min_i32 += 1;
        }
    } else {
        while counter > e_min && e_min_i32 > -1000 {
            counter = counter - one;
            e_min_i32 -= 1;
        }
    }

    let mut e_max_i32 = 0i32;
    let mut counter = zero;

    if e_max >= zero {
        while counter < e_max && e_max_i32 < 1000 {
            counter = counter + one;
            e_max_i32 += 1;
        }
    } else {
        while counter > e_max && e_max_i32 > -1000 {
            counter = counter - one;
            e_max_i32 -= 1;
        }
    }

    Some((e_min_i32, e_max_i32))
}

impl<D, N> Logarithmic<D, N>
where
    D: Float + 'static,
    N: Float + 'static,
{
    /// Creates a new logarithmic scale with the given base and domain range.
    ///
    /// Uses the default tick generator which creates ticks at powers of the base
    /// (e.g., for base 10: 1, 10, 100, 1000) with minor ticks in between.
    ///
    /// # Arguments
    ///
    /// * `base` - The logarithm base (typically 10 or e)
    /// * `min` - The minimum value of the domain (must be > 0)
    /// * `max` - The maximum value of the domain (must be > 0)
    ///
    /// # Examples
    ///
    /// ```
    /// use aksel::{Scale, scale::Logarithmic};
    ///
    /// // Base-10 logarithmic scale
    /// let scale = Logarithmic::<f64, f64>::new(10.0, 1.0, 1000.0);
    /// assert_eq!(scale.domain(), (&1.0, &1000.0));
    ///
    /// // Natural logarithm (base e)
    /// let e = std::f64::consts::E;
    /// let scale = Logarithmic::<f64, f64>::new(e, 1.0, 100.0);
    /// ```
    pub fn new(base: D, min: D, max: D) -> Self {
        Self::new_with_tick_generator(base, min, max, default_tick_generator)
    }

    /// Creates a new logarithmic scale with a custom tick generator.
    ///
    /// The tick generator is a function that takes a reference to the scale
    /// and returns a [`TickIter`] that generates tick marks.
    ///
    /// # Arguments
    ///
    /// * `base` - The logarithm base
    /// * `min` - The minimum value of the domain (must be > 0)
    /// * `max` - The maximum value of the domain (must be > 0)
    /// * `tick_generator` - A function that generates ticks for this scale
    ///
    /// # Examples
    ///
    /// ```
    /// use aksel::{Scale, scale::{Logarithmic, TickIter}};
    ///
    /// let scale = Logarithmic::<f64, f64>::new_with_tick_generator(
    ///     10.0, 1.0, 1000.0,
    ///     |_scale| TickIter::empty()
    /// );
    /// assert!(scale.ticks().is_empty());
    /// ```
    pub fn new_with_tick_generator<F>(base: D, min: D, max: D, tick_generator: F) -> Self
    where
        F: Fn(&Self) -> TickIter<D> + 'static,
    {
        Self {
            base,
            min,
            max,
            tick_generator: Box::new(tick_generator),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Creates a new logarithmic scale with a custom tick function.
    ///
    /// This is a convenience method that wraps a function returning `Vec<Tick<D>>`
    /// into a tick generator. Use this when you want to provide a simple list
    /// of ticks rather than implementing an iterator.
    ///
    /// # Arguments
    ///
    /// * `base` - The logarithm base
    /// * `min` - The minimum value of the domain (must be > 0)
    /// * `max` - The maximum value of the domain (must be > 0)
    /// * `tick_fn` - A function that generates a vector of ticks
    ///
    /// # Examples
    ///
    /// ```
    /// use aksel::{Scale, scale::{Logarithmic, Tick}};
    ///
    /// let scale = Logarithmic::<f64, f64>::new_with_tick_fn(
    ///     10.0, 1.0, 1000.0,
    ///     |_scale| vec![
    ///         Tick { value: 1.0, level: 0 },
    ///         Tick { value: 10.0, level: 0 },
    ///         Tick { value: 100.0, level: 0 },
    ///         Tick { value: 1000.0, level: 0 },
    ///     ]
    /// );
    /// assert_eq!(scale.ticks().len(), 4);
    /// ```
    pub fn new_with_tick_fn<F>(base: D, min: D, max: D, tick_fn: F) -> Self
    where
        F: Fn(&Self) -> Vec<Tick<D>> + 'static,
    {
        Self::new_with_tick_generator(base, min, max, move |scale| {
            TickIter::from_vec(tick_fn(scale))
        })
    }
}

fn default_tick_generator<D, N>(scale: &Logarithmic<D, N>) -> TickIter<D>
where
    D: Float + 'static,
    N: Float,
{
    TickIter::from_log(LogTickIter::from_scale(scale))
}

pub struct LogTickIter<D: Float> {
    state: LogTickIterState<D>,
}

enum LogTickIterState<D: Float> {
    Normal(LogNormalState<D>),
    Fallback(FallbackState<D>),
    Done,
}

struct LogNormalState<D: Float> {
    base: D,
    domain_min: D,
    domain_max: D,
    exponent: i32,
    exponent_max: i32,
    current_decade: D,
    multipliers: Vec<D>,
    multiplier_idx: usize,
    stage: LogStage,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum LogStage {
    Major,
    Minor,
}

struct FallbackState<D: Float> {
    first: Option<D>,
    second: Option<D>,
}

impl<D: Float> LogTickIter<D> {
    pub(crate) fn from_scale<N: Float>(scale: &Logarithmic<D, N>) -> Self {
        let (domain_min, domain_max) = util::sorted_pair(scale.min, scale.max);

        if let Some((e_min, e_max)) =
            log_ticks_exponent_range(&domain_min, &domain_max, &scale.base)
        {
            let mut multipliers = build_minor_multipliers(&scale.base);
            multipliers.shrink_to_fit();
            let current_decade = scale.base.powi(e_min);

            Self {
                state: LogTickIterState::Normal(LogNormalState {
                    base: scale.base,
                    domain_min,
                    domain_max,
                    exponent: e_min,
                    exponent_max: e_max,
                    current_decade,
                    multipliers,
                    multiplier_idx: 0,
                    stage: LogStage::Major,
                }),
            }
        } else {
            Self {
                state: LogTickIterState::Fallback(FallbackState {
                    first: Some(scale.min),
                    second: Some(scale.max),
                }),
            }
        }
    }
}

impl<D: Float> Iterator for LogTickIter<D> {
    type Item = Tick<D>;

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.state {
            LogTickIterState::Normal(state) => {
                if let Some(tick) = state.next_tick() {
                    Some(tick)
                } else {
                    self.state = LogTickIterState::Done;
                    None
                }
            }
            LogTickIterState::Fallback(state) => {
                if let Some(value) = state.first.take() {
                    Some(Tick { value, level: 0 })
                } else if let Some(value) = state.second.take() {
                    Some(Tick { value, level: 0 })
                } else {
                    self.state = LogTickIterState::Done;
                    None
                }
            }
            LogTickIterState::Done => None,
        }
    }
}

impl<D: Float> LogNormalState<D> {
    fn next_tick(&mut self) -> Option<Tick<D>> {
        loop {
            if self.exponent > self.exponent_max {
                return None;
            }

            match self.stage {
                LogStage::Major => {
                    let value = self.current_decade;

                    // Prepare next stage
                    if self.multipliers.is_empty() {
                        if !self.advance_decade() {
                            // Still return current value if it was valid
                            if value >= self.domain_min && value <= self.domain_max {
                                return Some(Tick { value, level: 0 });
                            } else {
                                return None;
                            }
                        }
                    } else {
                        self.stage = LogStage::Minor;
                        self.multiplier_idx = 0;
                    }

                    if value >= self.domain_min && value <= self.domain_max {
                        return Some(Tick { value, level: 0 });
                    }
                }
                LogStage::Minor => {
                    if self.multiplier_idx >= self.multipliers.len() {
                        if !self.advance_decade() {
                            return None;
                        }
                        self.stage = LogStage::Major;
                        continue;
                    }

                    let multiplier = self.multipliers[self.multiplier_idx];
                    self.multiplier_idx += 1;
                    let value = self.current_decade * multiplier;

                    if value >= self.domain_min && value <= self.domain_max {
                        return Some(Tick { value, level: 1 });
                    }
                }
            }
        }
    }

    fn advance_decade(&mut self) -> bool {
        self.exponent += 1;
        if self.exponent > self.exponent_max {
            false
        } else {
            self.current_decade = self.base.powi(self.exponent);
            true
        }
    }
}

fn build_minor_multipliers<D: Float>(base: &D) -> Vec<D> {
    let one = D::one();
    let floor = base.floor();
    let mut multipliers = Vec::new();

    let mut value = one + one;
    let mut guard = 0;

    while value < floor && guard < 100 {
        multipliers.push(value);
        value = value + one;
        guard += 1;
    }

    multipliers
}

impl<D, N> Scale for Logarithmic<D, N>
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
        if !self.is_valid_domain_value(value) {
            return Some(N::zero());
        }

        let ln_min = self.min.ln();
        let ln_max = self.max.ln();
        let span_d = ln_max - ln_min;
        if span_d == D::zero() {
            return Some(N::zero());
        }

        let ln_v = value.ln();
        let offset_d = ln_v - ln_min;

        // Convert to normalized type
        let offset_n: N = N::from(offset_d)?;
        let span_n: N = N::from(span_d)?;

        Some(offset_n / span_n)
    }

    fn denormalize_opt(&self, t: N) -> Option<D> {
        let ln_min = self.min.ln();
        let ln_max = self.max.ln();
        let span_d = ln_max - ln_min;
        let span_n: N = N::from(span_d)?;

        let scaled_n = t * span_n;
        let scaled_d: D = D::from(scaled_n)?;
        let ln_v = ln_min + scaled_d;

        Some(ln_v.exp())
    }

    fn pan_opt(&mut self, delta_norm: N) -> Option<()> {
        // Pan in logarithmic space, not linear domain space.
        // This means shifting the log(min) and log(max) by the same amount.

        let ln_min = self.min.ln();
        let ln_max = self.max.ln();

        // Calculate the span in log space
        let ln_span_d = ln_max - ln_min;
        let ln_span_n: N = N::from(ln_span_d)?;

        // Calculate the shift in log space
        let ln_shift_n = ln_span_n * delta_norm;
        let ln_shift: D = D::from(ln_shift_n)?;

        // Apply shift in log space, then convert back to domain
        let new_ln_min = ln_min + ln_shift;
        let new_ln_max = ln_max + ln_shift;

        self.min = new_ln_min.exp();
        self.max = new_ln_max.exp();
        Some(())
    }

    fn zoom_opt(&mut self, factor: N, anchor_norm: Option<N>) -> Option<()> {
        if factor <= N::zero() {
            return None;
        }

        let one = N::one();
        let two = one + one;
        let half = one / two;
        let anchor_norm = anchor_norm.unwrap_or(half);

        // Work in logarithmic space
        let ln_min = self.min.ln();
        let ln_max = self.max.ln();

        // Calculate span in log space
        let ln_span_d = ln_max - ln_min;
        let ln_span_n: N = N::from(ln_span_d)?;

        // New span in log space
        let new_ln_span_n = ln_span_n / factor;

        // Calculate anchor position in log space
        // The anchor is at: ln_min + anchor_norm * ln_span
        let anchor_offset_n = ln_span_n * anchor_norm;
        let anchor_offset_d: D = D::from(anchor_offset_n)?;
        let ln_anchor = ln_min + anchor_offset_d;

        // Split new span around anchor based on normalized anchor position
        let left_frac_n = anchor_norm;
        let right_frac_n = one - anchor_norm;

        let left_shift: D = D::from(new_ln_span_n * left_frac_n)?;
        let right_shift: D = D::from(new_ln_span_n * right_frac_n)?;

        let new_ln_min = ln_anchor - left_shift;
        let new_ln_max = ln_anchor + right_shift;

        // Convert back from log space to domain
        self.min = new_ln_min.exp();
        self.max = new_ln_max.exp();
        Some(())
    }

    fn tick_iter(&self) -> TickIter<D> {
        (self.tick_generator)(self)
    }

    fn extend_domain(&mut self, other_min: &D, other_max: &D) {
        // Only extend with valid (positive) values.
        if self.is_valid_domain_value(other_min) && other_min < &self.min {
            self.min = *other_min;
        }
        if self.is_valid_domain_value(other_max) && other_max > &self.max {
            self.max = *other_max;
        }
    }

    fn is_valid_domain_value(&self, value: &D) -> bool {
        // For logarithmic, only values > 0 are valid.
        *value > D::zero()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_normalize_base10() {
        let scale = Logarithmic::<f64, f64>::new(10.0, 1.0, 100.0);

        // log10(1) = 0, log10(100) = 2
        // Normalized range: [0, 2]
        assert_eq!(scale.normalize(&1.0), 0.0);
        assert_eq!(scale.normalize(&100.0), 1.0);

        // log10(10) = 1, which is halfway in log space
        let mid = scale.normalize(&10.0);
        assert!((mid - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_log_normalize_base_e() {
        let scale = Logarithmic::<f64, f64>::new(std::f64::consts::E, 1.0, std::f64::consts::E);

        // ln(1) = 0, ln(e) = 1
        assert_eq!(scale.normalize(&1.0), 0.0);
        assert!((scale.normalize(&std::f64::consts::E) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_log_denormalize() {
        let scale = Logarithmic::<f64, f64>::new(10.0, 1.0, 100.0);

        assert!((scale.denormalize(0.0) - 1.0).abs() < 1e-10);
        assert!((scale.denormalize(0.5) - 10.0).abs() < 1e-10);
        assert!((scale.denormalize(1.0) - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_log_normalize_f32() {
        let scale = Logarithmic::<f32, f32>::new(10.0, 1.0, 100.0);

        assert_eq!(scale.normalize(&1.0), 0.0);
        assert_eq!(scale.normalize(&100.0), 1.0);

        let mid = scale.normalize(&10.0);
        assert!((mid - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_log_invalid_values() {
        let scale = Logarithmic::<f64, f64>::new(10.0, 1.0, 100.0);

        // Zero and negative values should return 0.0
        assert_eq!(scale.normalize(&0.0), 0.0);
        assert_eq!(scale.normalize(&-10.0), 0.0);
    }

    #[test]
    fn test_log_is_valid_domain_value() {
        let scale = Logarithmic::<f64, f64>::new(10.0, 1.0, 100.0);

        assert!(scale.is_valid_domain_value(&1.0));
        assert!(scale.is_valid_domain_value(&10.0));
        assert!(!scale.is_valid_domain_value(&0.0));
        assert!(!scale.is_valid_domain_value(&-5.0));
    }

    #[test]
    fn test_log_pan() {
        let mut scale = Logarithmic::<f64, f64>::new(10.0, 1.0, 100.0);

        // Pan by 10% in normalized log space
        // Original: [1.0, 100.0] in domain, [0, 2] in log10 space
        // Log span = 2, shift = 0.1 * 2 = 0.2
        // New log range: [0.2, 2.2] → domain: [10^0.2, 10^2.2] ≈ [1.585, 158.49]
        scale.pan(0.1);

        let (min, max) = scale.domain();

        // After panning in log space, both min and max should increase
        assert!(*min > 1.0);
        assert!(*max > 100.0);

        // Check approximate values
        assert!((*min - 1.585).abs() < 0.01);
        assert!((*max - 158.49).abs() < 0.01);

        // The ratio should be preserved (geometric relationship)
        let original_ratio = 100.0 / 1.0;
        let new_ratio = *max / *min;
        assert!((new_ratio - original_ratio).abs() < 0.01);
    }

    #[test]
    fn test_log_zoom_in() {
        let mut scale = Logarithmic::<f64, f64>::new(10.0, 1.0, 100.0);

        // Zoom in by 2x at center (0.5 in normalized log space)
        // Original: [1.0, 100.0] in domain, [0, 2] in log10 space
        // Log span = 2, new log span = 2/2 = 1
        // Anchor at 0.5: log10(anchor) = 0 + 0.5 * 2 = 1, so anchor = 10
        // New log range: [1 - 0.5*1, 1 + 0.5*1] = [0.5, 1.5]
        // Domain range: [10^0.5, 10^1.5] ≈ [3.162, 31.623]
        scale.zoom(2.0, Some(0.5));

        let (min, max) = scale.domain();

        // Check that range is smaller
        assert!(*min > 1.0);
        assert!(*max < 100.0);

        // Check approximate values
        assert!((*min - 3.162).abs() < 0.01);
        assert!((*max - 31.623).abs() < 0.01);

        // The geometric center should be preserved
        let geometric_center = (*min * *max).sqrt();
        assert!((geometric_center - 10.0).abs() < 0.01);

        // The ratio should be sqrt of original (zoomed in by 2x)
        let original_ratio: f64 = 100.0 / 1.0; // = 100
        let new_ratio = *max / *min; // should be sqrt(100) = 10
        assert!((new_ratio - original_ratio.sqrt()).abs() < 0.01);
    }

    #[test]
    fn test_log_zoom_out() {
        let mut scale = Logarithmic::<f64, f64>::new(10.0, 10.0, 100.0);

        // Zoom out by 0.5x (makes range 2x larger) at center
        // Original: [10.0, 100.0] in domain, [1, 2] in log10 space
        // Log span = 1, new log span = 1/0.5 = 2
        // Anchor at 0.5: log10(anchor) = 1 + 0.5 * 1 = 1.5, so anchor = 10^1.5 ≈ 31.623
        // New log range: [1.5 - 0.5*2, 1.5 + 0.5*2] = [0.5, 2.5]
        // Domain range: [10^0.5, 10^2.5] ≈ [3.162, 316.23]
        scale.zoom(0.5, Some(0.5));

        let (min, max) = scale.domain();

        // Check that range is larger
        assert!(*min < 10.0);
        assert!(*max > 100.0);

        // Check approximate values
        assert!((*min - 3.162).abs() < 0.01);
        assert!((*max - 316.23).abs() < 0.01);

        // The geometric center should be preserved
        let geometric_center = (*min * *max).sqrt();
        let original_center = (10.0 * 100.0_f64).sqrt();
        assert!((geometric_center - original_center).abs() < 0.1);

        // The ratio should be square of original (zoomed out by 0.5x)
        let original_ratio: f64 = 100.0 / 10.0; // = 10
        let new_ratio = *max / *min; // should be 10^2 = 100
        assert!(original_ratio.mul_add(-original_ratio, new_ratio).abs() < 0.1);
    }

    #[test]
    fn test_log_ticks_base10() {
        let scale = Logarithmic::<f64, f64>::new(10.0, 1.0, 1000.0);
        let ticks = scale.ticks();

        assert!(!ticks.is_empty());

        // Should have major ticks at powers of 10
        let majors: Vec<_> = ticks.iter().filter(|t| t.level == 0).collect();
        assert!(!majors.is_empty());

        let major_values: Vec<_> = majors.iter().map(|t| t.value).collect();

        // Check for powers of 10
        assert!(major_values.iter().any(|&v| (v - 1.0).abs() < 1e-6));
        assert!(major_values.iter().any(|&v| (v - 10.0).abs() < 1e-6));
        assert!(major_values.iter().any(|&v| (v - 100.0).abs() < 1e-6));
        assert!(major_values.iter().any(|&v| (v - 1000.0).abs() < 1e-6));
    }

    #[test]
    fn test_log_ticks_with_minors() {
        let scale = Logarithmic::<f64, f64>::new(10.0, 1.0, 100.0);
        let ticks = scale.ticks();

        let minors: Vec<_> = ticks.iter().filter(|t| t.level == 1).collect();

        // Should have both major and minor ticks
        assert!(ticks.iter().any(|t| t.level == 0));
        assert!(!minors.is_empty());

        // Minors should include values like 2, 3, 4, ... times powers of 10
        let minor_values: Vec<_> = minors.iter().map(|t| t.value).collect();

        // Check for some expected minor ticks (e.g., 2, 3, 20, 30)
        assert!(minor_values.iter().any(|&v| (v - 2.0).abs() < 1e-6));
        assert!(minor_values.iter().any(|&v| (v - 20.0).abs() < 1e-6));
    }

    #[test]
    fn test_log_ticks_sorted() {
        let scale = Logarithmic::<f64, f64>::new(10.0, 1.0, 100.0);
        let ticks = scale.ticks();

        // Verify ticks are sorted
        for i in 1..ticks.len() {
            assert!(ticks[i].value >= ticks[i - 1].value);
        }
    }

    #[test]
    fn test_log_ticks_reversed_domain() {
        let scale = Logarithmic::<f64, f64>::new(10.0, 100.0, 1.0);
        let ticks = scale.ticks();

        assert!(!ticks.is_empty());

        let majors: Vec<_> = ticks.iter().filter(|t| t.level == 0).collect();
        assert!(majors.iter().any(|t| (t.value - 1.0).abs() < 1e-6));
        assert!(majors.iter().any(|t| (t.value - 10.0).abs() < 1e-6));
        assert!(majors.iter().any(|t| (t.value - 100.0).abs() < 1e-6));
    }

    #[test]
    fn test_log_custom_tick_generator() {
        let custom: Logarithmic<f64, f64> =
            Logarithmic::new_with_tick_fn(10.0f64, 1.0, 100.0, |scale| {
                vec![
                    Tick {
                        value: scale.min,
                        level: 0,
                    },
                    Tick {
                        value: scale.max,
                        level: 0,
                    },
                ]
            });

        let ticks = custom.ticks();
        assert_eq!(ticks.len(), 2);
        assert!((ticks[0].value - 1.0).abs() < 1e-6);
        assert!((ticks[1].value - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_log_extend_domain() {
        let mut scale = Logarithmic::<f64, f64>::new(10.0, 10.0, 100.0);

        // Extend to include 1.0..1000.0
        scale.extend_domain(&1.0, &1000.0);

        let (min, max) = scale.domain();
        assert_eq!(*min, 1.0);
        assert_eq!(*max, 1000.0);
    }

    #[test]
    fn test_log_extend_domain_invalid() {
        let mut scale = Logarithmic::<f64, f64>::new(10.0, 10.0, 100.0);

        // Try to extend with invalid values (should be ignored)
        scale.extend_domain(&0.0, &1000.0);

        let (min, max) = scale.domain();
        assert_eq!(*min, 10.0); // Should not change (0.0 is invalid)
        assert_eq!(*max, 1000.0); // Should extend to 1000.0
    }

    #[test]
    fn test_log_mixed_types() {
        // Domain is f64, Normalized is f32
        let scale = Logarithmic::<f64, f32>::new(10.0, 1.0, 100.0);

        let normalized: f32 = scale.normalize(&10.0);
        assert!((normalized - 0.5).abs() < 1e-6);

        let denormalized: f64 = scale.denormalize(0.5f32);
        assert!((denormalized - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_log_ticks_remain_within_domain() {
        let scale = Logarithmic::<f64, f64>::new(10.0, 1.3, 347.0);
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
    fn test_log_ticks_do_not_overlap_levels() {
        let scale = Logarithmic::<f64, f64>::new(10.0, 1.3, 347.0);
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
}
