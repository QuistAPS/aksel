//! Coordinate transformations between screen space and plot space.
//!
//! This module provides types and transformations for converting between:
//! - **Screen space**: Pixel coordinates on the display
//! - **Plot/Chart space**: Data values in your domain
//!
//! # Overview
//!
//! The [`Transform`] type connects scales to screen rectangles, handling
//! the mathematics of converting data coordinates to pixel positions and
//! vice versa.
//!
//! # Key Types
//!
//! - [`ScreenRect`] - A rectangle in pixel/screen coordinates
//! - [`ScreenPoint`] - A point in pixel/screen coordinates
//! - [`PlotRect`] - A rectangle in data/chart coordinates
//! - [`PlotPoint`] - A point in data/chart coordinates
//! - [`Transform`] - Converts between screen and plot coordinates
//!
//! # Coordinate Systems
//!
//! ## Screen Coordinates
//!
//! - Origin (0, 0) is typically at the top-left
//! - X increases to the right
//! - Y increases downward
//! - Values are in pixels
//!
//! ## Plot/Chart Coordinates
//!
//! - Origin and scale defined by your data
//! - X increases to the right
//! - Y increases upward (note the difference!)
//! - Values are in your domain units
//!
//! The [`Transform`] automatically handles the Y-axis inversion between
//! these coordinate systems.
//!
//! # Examples
//!
//! ## Basic Point Transformation
//!
//! ```rust
//! use aksel::{Transform, scale::Linear, ScreenRect, ScreenPoint, PlotPoint};
//!
//! // Define screen area (800x600 pixels)
//! let screen = ScreenRect {
//!     x: 0.0,
//!     y: 0.0,
//!     width: 800.0,
//!     height: 600.0,
//! };
//!
//! // Define scales for x and y axes
//! let x_scale = Linear::<f64, f32>::new(0.0, 100.0);
//! let y_scale = Linear::<f64, f32>::new(0.0, 50.0);
//!
//! // Create transform
//! let transform = Transform::new(&screen, &x_scale, &y_scale);
//!
//! // Convert plot point to screen coordinates
//! let plot_point = PlotPoint::new(50.0, 25.0); // Center of data
//! let screen_point = transform.chart_to_screen(&plot_point);
//! // Result: (400.0, 300.0) - center of screen
//! ```
//!
//! ## Rectangle Transformation
//!
//! ```rust
//! use aksel::{Transform, scale::Linear, ScreenRect, PlotRect};
//!
//! let screen = ScreenRect { x: 0.0, y: 0.0, width: 800.0, height: 600.0 };
//! let x_scale = Linear::<f64, f32>::new(0.0, 100.0);
//! let y_scale = Linear::<f64, f32>::new(0.0, 50.0);
//! let transform = Transform::new(&screen, &x_scale, &y_scale);
//!
//! // Define a selection rectangle in plot space
//! let plot_rect = PlotRect {
//!     x: 25.0,
//!     y: 12.5,
//!     width: 50.0,
//!     height: 25.0,
//! };
//!
//! // Convert to screen space
//! let screen_rect = transform.chart_to_screen_rect(plot_rect);
//! ```
//!
//! ## Handling User Input
//!
//! ```rust
//! use aksel::{Transform, scale::Linear, ScreenRect, ScreenPoint};
//!
//! let screen = ScreenRect { x: 0.0, y: 0.0, width: 800.0, height: 600.0 };
//! let x_scale = Linear::<f64, f32>::new(0.0, 100.0);
//! let y_scale = Linear::<f64, f32>::new(0.0, 50.0);
//! let transform = Transform::new(&screen, &x_scale, &y_scale);
//!
//! // User clicks at pixel (200, 450)
//! let click = ScreenPoint::new(200.0, 450.0);
//!
//! // Convert to data coordinates
//! let data_point = transform.screen_to_chart(&click);
//! // Now you know what data the user clicked on
//! ```
//!
//! ## Y-Axis Inversion
//!
//! The transform automatically handles the fact that screen Y increases
//! downward while chart Y typically increases upward:
//!
//! ```rust
//! use aksel::{Transform, scale::Linear, ScreenRect, PlotPoint};
//!
//! let screen = ScreenRect { x: 0.0, y: 0.0, width: 800.0, height: 600.0 };
//! let x_scale = Linear::<f64, f32>::new(0.0, 100.0);
//! let y_scale = Linear::<f64, f32>::new(0.0, 50.0);
//! let transform = Transform::new(&screen, &x_scale, &y_scale);
//!
//! // Bottom of chart (y=0) maps to bottom of screen (y=600)
//! let bottom = PlotPoint::new(0.0, 0.0);
//! let screen_bottom = transform.chart_to_screen(&bottom);
//! assert_eq!(screen_bottom.y, 600.0);
//!
//! // Top of chart (y=50) maps to top of screen (y=0)
//! let top = PlotPoint::new(0.0, 50.0);
//! let screen_top = transform.chart_to_screen(&top);
//! assert_eq!(screen_top.y, 0.0);
//! ```

use num_traits::Float;

use crate::scale::{Scale, util::sorted_pair};

/// A rectangle in screen/pixel coordinates.
///
/// Represents a rectangular region on the display, with position and size
/// measured in pixels.
///
/// # Fields
///
/// - `x`: X coordinate of the top-left corner
/// - `y`: Y coordinate of the top-left corner
/// - `width`: Width in pixels
/// - `height`: Height in pixels
#[derive(Debug, Clone, Copy)]
pub struct ScreenRect<S = f32> {
    /// X coordinate of the top-left corner in pixels.
    pub x: S,
    /// Y coordinate of the top-left corner in pixels.
    pub y: S,
    /// Width of the rectangle in pixels.
    pub width: S,
    /// Height of the rectangle in pixels.
    pub height: S,
}

/// A point in screen/pixel coordinates.
///
/// Represents a 2D position on the display measured in pixels.
#[derive(Debug, Clone, Copy)]
pub struct ScreenPoint<S = f32> {
    /// X coordinate in pixels.
    pub x: S,
    /// Y coordinate in pixels.
    pub y: S,
}

impl<S> ScreenPoint<S> {
    /// Creates a new screen point at the given pixel coordinates.
    pub const fn new(x: S, y: S) -> Self {
        Self { x, y }
    }
}

/// A point in plot/chart coordinates.
///
/// Represents a 2D position in data/chart space using domain values.
#[derive(Debug, Clone, Copy, Default)]
pub struct PlotPoint<D = f64> {
    /// X coordinate in domain units.
    pub x: D,
    /// Y coordinate in domain units.
    pub y: D,
}

impl<D> PlotPoint<D> {
    /// Creates a new plot point at the given data coordinates.
    pub const fn new(x: D, y: D) -> Self {
        Self { x, y }
    }
}

/// A rectangle in plot/chart coordinates.
///
/// Represents a rectangular region in data/chart space.
///
/// # Fields
///
/// - `x`: X coordinate of the bottom-left corner (minimum X)
/// - `y`: Y coordinate of the bottom-left corner (minimum Y)
/// - `width`: Width in domain units (can be negative)
/// - `height`: Height in domain units (can be negative)
///
/// Negative widths and heights are handled correctly during transformations.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PlotRect<D = f64> {
    /// X coordinate of the rectangle's origin in domain units.
    pub x: D,
    /// Y coordinate of the rectangle's origin in domain units.
    pub y: D,
    /// Width of the rectangle in domain units (can be negative).
    pub width: D,
    /// Height of the rectangle in domain units (can be negative).
    pub height: D,
}

impl<D: Copy> PlotRect<D> {
    /// Returns the minimum X coordinate (left edge) of the rectangle.
    pub const fn min_x(&self) -> D {
        self.x
    }

    /// Returns the minimum Y coordinate (bottom edge) of the rectangle.
    pub const fn min_y(&self) -> D {
        self.y
    }
}

impl<D: Float> PlotRect<D> {
    /// Creates a `PlotRect` from two opposite corner points.
    ///
    /// The resulting rectangle will have positive width and height,
    /// regardless of the order of the points.
    ///
    /// # Examples
    ///
    /// ```
    /// use aksel::{PlotPoint, PlotRect};
    ///
    /// let p1 = PlotPoint::new(10.0, 20.0);
    /// let p2 = PlotPoint::new(50.0, 80.0);
    ///
    /// let rect = PlotRect::from_points(p1, p2);
    /// assert_eq!(rect.x, 10.0);
    /// assert_eq!(rect.y, 20.0);
    /// assert_eq!(rect.width, 40.0);
    /// assert_eq!(rect.height, 60.0);
    /// ```
    pub fn from_points(p1: PlotPoint<D>, p2: PlotPoint<D>) -> Self {
        // Find the minimum and maximum X and Y coordinates using sorted_pair
        let (x_min, x_max) = sorted_pair(p1.x, p2.x);
        let (y_min, y_max) = sorted_pair(p1.y, p2.y);

        // Calculate the positive width and height
        let width = x_max - x_min;
        let height = y_max - y_min;

        Self {
            x: x_min,
            y: y_min,
            width,
            height,
        }
    }

    /// Creates a rectangle centered at the given point with the specified width and height.
    ///
    /// # Arguments
    ///
    /// * `center` - The center point of the rectangle.
    /// * `width` - The width of the rectangle.
    /// * `height` - The height of the rectangle.
    ///
    /// # Returns
    ///
    /// A new `PlotRect` instance.
    ///
    /// # Examples
    ///
    /// ```
    /// use aksel::{PlotPoint, PlotRect};
    ///
    /// let center = PlotPoint::new(50.0, 50.0);
    /// let width = 100.0;
    /// let height = 200.0;
    ///
    /// let rect = PlotRect::from_center(center, width, height);
    /// assert_eq!(rect.x, 25.0);
    /// assert_eq!(rect.y, 0.0);
    /// assert_eq!(rect.width, 100.0);
    /// assert_eq!(rect.height, 200.0);
    /// ```

    pub fn from_center(center: PlotPoint<D>, width: D, height: D) -> Self {
        let half_width = width / D::from(2).unwrap();
        let half_height = height / D::from(2).unwrap();

        Self {
            x: center.x - half_width,
            y: center.y - half_height,
            width,
            height,
        }
    }

    /// Returns the maximum X coordinate (right edge) of the rectangle.
    ///
    /// Computed as `x + width`.
    pub fn max_x(&self) -> D {
        self.x + self.width
    }

    /// Returns the maximum Y coordinate (top edge) of the rectangle.
    ///
    /// Computed as `y + height`.
    pub fn max_y(&self) -> D {
        self.y + self.height
    }

    /// Returns true if the provided point lies within this rectangle.
    ///
    /// Bounds are inclusive and negative spans are handled correctly.
    pub fn contains(&self, point: &PlotPoint<D>) -> bool {
        self.contains_x(&point.x) && self.contains_y(&point.y)
    }

    /// Returns true if the provided X value lies within the horizontal extent.
    ///
    /// Works for rectangles with negative widths by comparing sorted endpoints.
    pub fn contains_x(&self, value: &D) -> bool {
        let (min_x, max_x) = sorted_pair(self.x, self.x + self.width);
        value >= &min_x && value <= &max_x
    }

    /// Returns true if the provided Y value lies within the vertical extent.
    ///
    /// Works for rectangles with negative heights by comparing sorted endpoints.
    pub fn contains_y(&self, value: &D) -> bool {
        let (min_y, max_y) = sorted_pair(self.y, self.y + self.height);
        value >= &min_y && value <= &max_y
    }
}

#[cfg(test)]
mod plot_rect_tests {
    use super::{PlotPoint, PlotRect};

    #[test]
    fn contains_point_in_positive_rect() {
        let rect = PlotRect {
            x: 0.0f64,
            y: 0.0f64,
            width: 10.0,
            height: 5.0,
        };

        assert!(rect.contains(&PlotPoint::new(5.0, 3.0)));
        assert!(rect.contains_x(&0.0));
        assert!(rect.contains_y(&5.0));
    }

    #[test]
    fn contains_handles_negative_spans() {
        let rect = PlotRect {
            x: 10.0f64,
            y: 2.0f64,
            width: -4.0,
            height: -6.0,
        };

        assert!(rect.contains(&PlotPoint::new(8.0, -1.0)));
        assert!(rect.contains_x(&6.0));
        assert!(rect.contains_y(&2.0));
    }

    #[test]
    fn contains_rejects_outside_values() {
        let rect = PlotRect {
            x: -5.0f64,
            y: -5.0f64,
            width: 2.0,
            height: 2.0,
        };

        assert!(!rect.contains(&PlotPoint::new(-10.0, 0.0)));
        assert!(!rect.contains_x(&0.0));
        assert!(!rect.contains_y(&10.0));
    }
}

/// Transforms coordinates between screen space and plot/chart space.
///
/// `Transform` connects scales to screen rectangles, enabling conversion between
/// pixel coordinates and data values. It handles all the mathematics of coordinate
/// transformation, including y-axis inversion.
///
/// # Type Parameters
///
/// - `D`: Domain type (data values, typically `f64`)
/// - `N`: Normalized type (typically `f32`, used internally)
/// - `S`: Screen type (pixel coordinates, typically `f32`)
///
/// # Coordinate Spaces
///
/// The transform manages three coordinate spaces:
///
/// 1. **Domain/Plot Space**: Your data values (e.g., temperature, time, price)
/// 2. **Normalized Space**: Internal [0, 1] range (handled by scales)
/// 3. **Screen Space**: Pixel coordinates on the display
///
/// # Y-Axis Inversion
///
/// The transform automatically handles the fact that:
/// - Screen Y increases **downward** (top = 0, bottom = height)
/// - Chart Y typically increases **upward** (bottom = 0, top = max)
///
/// This inversion is built into all Y-axis transformations.
///
/// # Method Variants
///
/// Most methods come in two variants:
/// - `method()` - Panics on conversion failure
/// - `method_opt()` - Returns `Option`, safer for untrusted input
///
/// # Examples
///
/// ## Basic Point Transformation
///
/// ```rust
/// use aksel::{Transform, scale::Linear, ScreenRect, PlotPoint, ScreenPoint};
///
/// let screen = ScreenRect { x: 0.0, y: 0.0, width: 800.0, height: 600.0 };
/// let x_scale = Linear::<f64, f32>::new(0.0, 100.0);
/// let y_scale = Linear::<f64, f32>::new(0.0, 50.0);
/// let transform = Transform::new(&screen, &x_scale, &y_scale);
///
/// // Plot to screen
/// let plot = PlotPoint::new(50.0, 25.0); // Center of data
/// let screen_pt = transform.chart_to_screen(&plot);
/// assert_eq!(screen_pt.x, 400.0); // Center of screen X
/// assert_eq!(screen_pt.y, 300.0); // Center of screen Y
///
/// // Screen to plot
/// let screen = ScreenPoint::new(400.0, 300.0);
/// let plot_pt = transform.screen_to_chart(&screen);
/// assert_eq!(plot_pt.x, 50.0);
/// assert_eq!(plot_pt.y, 25.0);
/// ```
///
/// ## Individual Coordinate Transformation
///
/// ```rust
/// use aksel::{Transform, scale::Linear, ScreenRect};
///
/// let screen = ScreenRect { x: 0.0, y: 0.0, width: 800.0, height: 600.0 };
/// let x_scale = Linear::<f64, f32>::new(0.0, 100.0);
/// let y_scale = Linear::<f64, f32>::new(0.0, 50.0);
/// let transform = Transform::new(&screen, &x_scale, &y_scale);
///
/// // Transform individual coordinates
/// let screen_x = transform.x_to_screen(&75.0);
/// assert_eq!(screen_x, 600.0); // 75% of 800
///
/// let plot_y = transform.y_from_screen(&150.0);
/// assert_eq!(plot_y, 37.5); // accounting for Y inversion
/// ```
///
/// ## Rectangle Transformation
///
/// ```rust
/// use aksel::{Transform, scale::Linear, ScreenRect, PlotRect};
///
/// let screen = ScreenRect { x: 0.0, y: 0.0, width: 800.0, height: 600.0 };
/// let x_scale = Linear::<f64, f32>::new(0.0, 100.0);
/// let y_scale = Linear::<f64, f32>::new(0.0, 50.0);
/// let transform = Transform::new(&screen, &x_scale, &y_scale);
///
/// // Transform a rectangle from plot to screen
/// let plot_rect = PlotRect {
///     x: 25.0,
///     y: 12.5,
///     width: 50.0,
///     height: 25.0,
/// };
/// let screen_rect = transform.chart_to_screen_rect(plot_rect);
///
/// // Transform back
/// let plot_rect2 = transform.screen_to_chart_rect(screen_rect);
/// ```
///
/// ## Handling User Interaction
///
/// ```rust
/// use aksel::{Transform, scale::Linear, ScreenRect, ScreenPoint};
///
/// let screen = ScreenRect { x: 0.0, y: 0.0, width: 800.0, height: 600.0 };
/// let x_scale = Linear::<f64, f32>::new(0.0, 100.0);
/// let y_scale = Linear::<f64, f32>::new(0.0, 50.0);
/// let transform = Transform::new(&screen, &x_scale, &y_scale);
///
/// // User clicks at pixel (200, 450)
/// let click = ScreenPoint::new(200.0, 450.0);
/// let data_point = transform.screen_to_chart(&click);
///
/// // Now you know what data the user clicked on
/// println!("Clicked on data point: ({}, {})", data_point.x, data_point.y);
/// ```
///
/// ## Using with Logarithmic Scales
///
/// ```rust
/// use aksel::{Transform, scale::Logarithmic, ScreenRect, PlotPoint};
///
/// let screen = ScreenRect { x: 0.0, y: 0.0, width: 800.0, height: 600.0 };
/// let x_scale = Logarithmic::<f64, f32>::new(10.0, 1.0, 1000.0);
/// let y_scale = Logarithmic::<f64, f32>::new(10.0, 1.0, 100.0);
/// let transform = Transform::new(&screen, &x_scale, &y_scale);
///
/// // Logarithmic scales give equal screen space to equal ratios
/// let point1 = PlotPoint::new(10.0, 10.0);
/// let point2 = PlotPoint::new(100.0, 10.0);
///
/// let screen1 = transform.chart_to_screen(&point1);
/// let screen2 = transform.chart_to_screen(&point2);
///
/// // 1→10 and 10→100 have equal visual distance (both 10x)
/// ```
///
/// ## Safe Conversion with _opt Methods
///
/// ```rust
/// use aksel::{Transform, scale::Linear, ScreenRect, PlotPoint};
///
/// let screen = ScreenRect { x: 0.0, y: 0.0, width: 800.0, height: 600.0 };
/// let x_scale = Linear::<f64, f32>::new(0.0, 100.0);
/// let y_scale = Linear::<f64, f32>::new(0.0, 50.0);
/// let transform = Transform::new(&screen, &x_scale, &y_scale);
///
/// // Use _opt for safe conversion
/// let plot = PlotPoint::new(50.0, 25.0);
/// match transform.chart_to_screen_opt(&plot) {
///     Some(screen_pt) => println!("Converted to ({}, {})", screen_pt.x, screen_pt.y),
///     None => println!("Conversion failed"),
/// }
/// ```
#[derive(Clone, Copy)]
pub struct Transform<'a, D = f64, N = f32, S = f32> {
    screen_rect: &'a ScreenRect<S>,
    x_scale: &'a dyn Scale<Domain = D, Normalized = N>,
    y_scale: &'a dyn Scale<Domain = D, Normalized = N>,
}

impl<'a, D, N, S> Transform<'a, D, N, S> {
    /// Creates a new Transform mapping a screen rectangle to a chart rectangle.
    pub const fn new(
        screen_rect: &'a ScreenRect<S>,
        x_scale: &'a dyn Scale<Domain = D, Normalized = N>,
        y_scale: &'a dyn Scale<Domain = D, Normalized = N>,
    ) -> Self {
        Self {
            screen_rect,
            x_scale,
            y_scale,
        }
    }

    /// Returns the screen rectangle bounds used by this transform.
    ///
    /// This is the same rectangle that was passed to [`Transform::new`].
    ///
    /// # Examples
    ///
    /// ```
    /// use aksel::{Transform, scale::Linear, ScreenRect};
    ///
    /// let screen = ScreenRect { x: 0.0, y: 0.0, width: 800.0, height: 600.0 };
    /// let x_scale = Linear::<f64, f32>::new(0.0, 100.0);
    /// let y_scale = Linear::<f64, f32>::new(0.0, 50.0);
    /// let transform = Transform::new(&screen, &x_scale, &y_scale);
    ///
    /// let bounds = transform.screen_bounds();
    /// assert_eq!(bounds.width, 800.0);
    /// assert_eq!(bounds.height, 600.0);
    /// ```
    pub const fn screen_bounds(&self) -> &ScreenRect<S> {
        self.screen_rect
    }
}

/// Implementation of the transformation logic.
impl<'a, D, N, S> Transform<'a, D, N, S>
where
    N: Float,
    S: Float,
    D: Float,
{
    pub fn plot_bounds(&self) -> PlotRect<D> {
        let (&x_min, &x_max) = self.x_scale.domain();
        let (&y_min, &y_max) = self.y_scale.domain();
        PlotRect {
            x: x_min,
            y: y_min,
            width: (x_min - x_max).abs(),
            height: (y_min - y_max).abs(),
        }
    }

    /// Transforms a point from screen coordinates to chart coordinates.
    pub fn screen_to_chart_opt(&self, screen_point: &ScreenPoint<S>) -> Option<PlotPoint<D>> {
        let cx = self.x_from_screen_opt(&screen_point.x)?;
        let cy = self.y_from_screen_opt(&screen_point.y)?;

        Some(PlotPoint::new(cx, cy))
    }

    /// Transforms a point from screen coordinates to chart coordinates.
    pub fn screen_to_chart(&self, screen_point: &ScreenPoint<S>) -> PlotPoint<D> {
        self.screen_to_chart_opt(screen_point).unwrap()
    }

    /// Transforms a point from chart coordinates to screen coordinates.
    pub fn chart_to_screen_opt(&self, plot_point: &PlotPoint<D>) -> Option<ScreenPoint<S>> {
        let sx = self.x_to_screen_opt(&plot_point.x)?;
        let sy = self.y_to_screen_opt(&plot_point.y)?;

        Some(ScreenPoint::new(sx, sy))
    }

    /// Transforms a point from chart coordinates to screen coordinates.
    pub fn chart_to_screen(&self, plot_point: &PlotPoint<D>) -> ScreenPoint<S> {
        self.chart_to_screen_opt(plot_point).unwrap()
    }

    /// Transforms a single chart x-coordinate to a screen x-coordinate.
    pub fn x_to_screen_opt(&self, plot_x: &D) -> Option<S> {
        let norm_x: N = self.x_scale.normalize_opt(plot_x)?;
        let screen_x: S = S::from(norm_x)?;
        Some(self.screen_rect.x + screen_x * self.screen_rect.width)
    }

    /// Transforms a single chart x-coordinate to a screen x-coordinate.
    pub fn x_to_screen(&self, plot_x: &D) -> S {
        self.x_to_screen_opt(plot_x).unwrap()
    }

    /// Transforms a single chart y-coordinate to a screen y-coordinate.
    /// (This includes the y-axis inversion)
    pub fn y_to_screen_opt(&self, plot_y: &D) -> Option<S> {
        let norm_y: N = self.y_scale.normalize_opt(plot_y)?;
        let screen_norm_y: S = S::from(norm_y)?;

        // Invert Y: (1 - norm_y)
        let inverted = S::one() - screen_norm_y;

        Some(self.screen_rect.y + inverted * self.screen_rect.height)
    }

    /// Transforms a single chart y-coordinate to a screen y-coordinate.
    /// (This includes the y-axis inversion)
    pub fn y_to_screen(&self, plot_y: &D) -> S {
        self.y_to_screen_opt(plot_y).unwrap()
    }

    /// Transforms a single screen x-coordinate to a chart x-coordinate.
    pub fn x_from_screen_opt(&self, screen_x: &S) -> Option<D> {
        let norm_x_screen = (*screen_x - self.screen_rect.x) / self.screen_rect.width;
        let norm_x: N = N::from(norm_x_screen)?;
        self.x_scale.denormalize_opt(norm_x)
    }

    /// Transforms a single screen x-coordinate to a chart x-coordinate.
    pub fn x_from_screen(&self, screen_x: &S) -> D {
        self.x_from_screen_opt(screen_x).unwrap()
    }

    /// Transforms a single screen y-coordinate to a chart y-coordinate.
    /// (This accounts for the y-axis inversion)
    pub fn y_from_screen_opt(&self, screen_y: &S) -> Option<D> {
        let norm_y_raw_screen = (*screen_y - self.screen_rect.y) / self.screen_rect.height;

        // Invert the normalized screen Y: (1 - norm_y_raw)
        let norm_y_screen = S::one() - norm_y_raw_screen;

        let norm_y: N = N::from(norm_y_screen)?;
        self.y_scale.denormalize_opt(norm_y)
    }

    /// Transforms a single screen y-coordinate to a chart y-coordinate.
    /// (This accounts for the y-axis inversion)
    pub fn y_from_screen(&self, screen_y: &S) -> D {
        self.y_from_screen_opt(screen_y).unwrap()
    }

    // --- UPDATED: Rect transformations (using new helpers) ---

    /// Transforms a rectangle from screen coordinates to chart coordinates.
    pub fn screen_to_chart_rect_opt(&self, screen_rect: ScreenRect<S>) -> Option<PlotRect<D>> {
        let top_left_screen = ScreenPoint::new(screen_rect.x, screen_rect.y);
        let bottom_right_screen = ScreenPoint::new(
            screen_rect.x + screen_rect.width,
            screen_rect.y + screen_rect.height,
        );

        let top_left_chart = self.screen_to_chart_opt(&top_left_screen)?;
        let bottom_right_chart = self.screen_to_chart_opt(&bottom_right_screen)?;

        let (x_min_chart, x_max_chart) = if top_left_chart.x <= bottom_right_chart.x {
            (top_left_chart.x, bottom_right_chart.x)
        } else {
            (bottom_right_chart.x, top_left_chart.x)
        };

        let (y_min_chart, y_max_chart) = if top_left_chart.y <= bottom_right_chart.y {
            (top_left_chart.y, bottom_right_chart.y)
        } else {
            (bottom_right_chart.y, top_left_chart.y)
        };

        Some(PlotRect {
            x: x_min_chart,
            y: y_min_chart,
            width: x_max_chart - x_min_chart,
            height: y_max_chart - y_min_chart,
        })
    }

    /// Transforms a rectangle from screen coordinates to chart coordinates.
    pub fn screen_to_chart_rect(&self, screen_rect: ScreenRect<S>) -> PlotRect<D> {
        self.screen_to_chart_rect_opt(screen_rect).unwrap()
    }

    /// Transforms a rectangle from chart coordinates to screen coordinates.
    pub fn chart_to_screen_rect_opt(&self, plot_rect: PlotRect<D>) -> Option<ScreenRect<S>> {
        let x_end = plot_rect.x + plot_rect.width;
        let y_end = plot_rect.y + plot_rect.height;

        // Normalize both x endpoints so negative widths are handled by swapping
        let x_start_norm = self.x_scale.normalize_opt(&plot_rect.x)?;
        let x_end_norm = self.x_scale.normalize_opt(&x_end)?;

        let (left_norm, right_norm) = if x_start_norm <= x_end_norm {
            (x_start_norm, x_end_norm)
        } else {
            (x_end_norm, x_start_norm)
        };

        let width_norm = right_norm - left_norm;

        // Normalize y endpoints separately and then account for the inverted y-axis
        let y_start_norm = self.y_scale.normalize_opt(&plot_rect.y)?;
        let y_end_norm = self.y_scale.normalize_opt(&y_end)?;

        let (bottom_norm, top_norm) = if y_start_norm <= y_end_norm {
            (y_start_norm, y_end_norm)
        } else {
            (y_end_norm, y_start_norm)
        };

        let height_norm = top_norm - bottom_norm;

        let screen_x = self.screen_rect.x + S::from(left_norm)? * self.screen_rect.width;
        let screen_width = S::from(width_norm)? * self.screen_rect.width;

        let screen_y =
            self.screen_rect.y + (S::one() - S::from(top_norm)?) * self.screen_rect.height;
        let screen_height = S::from(height_norm)? * self.screen_rect.height;

        Some(ScreenRect {
            x: screen_x,
            y: screen_y,
            width: screen_width,
            height: screen_height,
        })
    }

    /// Transforms a rectangle from chart coordinates to screen coordinates.
    pub fn chart_to_screen_rect(&self, plot_rect: PlotRect<D>) -> ScreenRect<S> {
        self.chart_to_screen_rect_opt(plot_rect).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scale::Linear;

    #[test]
    fn chart_rect_to_screen_rect_positive_spans() {
        let x_scale = Linear::<f64, f32>::new(100.0f64, 200.0);
        let y_scale = Linear::<f64, f32>::new(-50.0f64, 50.0);
        let transform = Transform::new(
            &ScreenRect {
                x: 10.0f32,
                y: 20.0f32,
                width: 800.0f32,
                height: 400.0f32,
            },
            &x_scale,
            &y_scale,
        );

        let plot_rect = PlotRect {
            x: 120.0,
            y: -10.0,
            width: 30.0,
            height: 20.0,
        };

        let screen_rect = transform.chart_to_screen_rect(plot_rect);

        assert!((screen_rect.x - 170.0).abs() < 1e-4);
        assert!((screen_rect.width - 240.0).abs() < 1e-4);
        assert!((screen_rect.y - 180.0).abs() < 1e-4);
        assert!((screen_rect.height - 80.0).abs() < 1e-4);
    }

    #[test]
    fn chart_rect_to_screen_rect_negative_spans() {
        let x_scale = Linear::<f64, f32>::new(100.0f64, 200.0);
        let y_scale = Linear::<f64, f32>::new(-50.0f64, 50.0);
        let transform = Transform::new(
            &ScreenRect {
                x: 10.0f32,
                y: 20.0f32,
                width: 800.0f32,
                height: 400.0f32,
            },
            &x_scale,
            &y_scale,
        );

        let plot_rect = PlotRect {
            x: 180.0,
            y: 30.0,
            width: -40.0,
            height: -20.0,
        };

        let screen_rect = transform.chart_to_screen_rect(plot_rect);

        assert!((screen_rect.x - 330.0).abs() < 1e-4);
        assert!((screen_rect.width - 320.0).abs() < 1e-4);
        assert!((screen_rect.y - 100.0).abs() < 1e-4);
        assert!((screen_rect.height - 80.0).abs() < 1e-4);
    }

    #[test]
    fn screen_rect_to_chart_rect_positive_spans() {
        let x_scale = Linear::<f64, f32>::new(100.0f64, 200.0);
        let y_scale = Linear::<f64, f32>::new(-50.0f64, 50.0);
        let transform = Transform::new(
            &ScreenRect {
                x: 10.0f32,
                y: 20.0f32,
                width: 800.0f32,
                height: 400.0f32,
            },
            &x_scale,
            &y_scale,
        );

        let selected = ScreenRect {
            x: 210.0,
            y: 70.0,
            width: 200.0,
            height: 100.0,
        };

        let chart_rect = transform.screen_to_chart_rect(selected);

        assert!((chart_rect.x - 125.0).abs() < 1e-8);
        assert!((chart_rect.width - 25.0).abs() < 1e-8);
        assert!((chart_rect.y - 12.5).abs() < 1e-8);
        assert!((chart_rect.height - 25.0).abs() < 1e-8);
    }

    #[test]
    fn screen_rect_to_chart_rect_negative_spans() {
        let x_scale = Linear::<f64, f32>::new(100.0f64, 200.0);
        let y_scale = Linear::<f64, f32>::new(-50.0f64, 50.0);
        let transform = Transform::new(
            &ScreenRect {
                x: 10.0f32,
                y: 20.0f32,
                width: 800.0f32,
                height: 400.0f32,
            },
            &x_scale,
            &y_scale,
        );

        let selected = ScreenRect {
            x: 500.0,
            y: 240.0,
            width: -100.0,
            height: -120.0,
        };

        let chart_rect = transform.screen_to_chart_rect(selected);

        assert!((chart_rect.x - 148.75).abs() < 1e-6);
        assert!((chart_rect.width - 12.5).abs() < 1e-6);
        assert!((chart_rect.y + 5.0).abs() < 1e-6);
        assert!((chart_rect.height - 30.0).abs() < 1e-6);
    }
}
