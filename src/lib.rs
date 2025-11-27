//! Tjarting (Charting) library
//!
//! `tjart` is a charting library that provides coordinate transformations and scale mappings
//! for data visualization. It focuses on the mathematical foundations of charting:
//! mapping data values to screen coordinates and generating axis tick marks.
//!
//! # Core Concepts
//!
//! ## Scales
//!
//! Scales map data values (domain) to a normalized [0, 1] range. They support:
//! - Linear and logarithmic mappings
//! - Pan and zoom operations
//! - Automatic tick generation for axis labels
//! - Bidirectional mapping (normalize and denormalize)
//!
//! Available scale types:
//! - [`scale::Linear`] - Affine mapping for linear data
//! - [`scale::Logarithmic`] - Logarithmic mapping for exponential data
//!
//! ## Transforms
//!
//! Transforms connect scales to screen coordinates, converting between:
//! - [`PlotPoint`] - Data values in chart space
//! - [`ScreenPoint`] - Pixel coordinates in screen space
//!
//! The [`Transform`] type handles y-axis inversion (screen coordinates typically
//! increase downward, while chart coordinates increase upward).
//!
//! # Examples
//!
//! ## Basic Linear Scale
//!
//! ```rust
//! use tjart::{Scale, scale::Linear};
//!
//! // Create a scale mapping [0.0, 100.0] to [0.0, 1.0]
//! let scale = Linear::<f64, f64>::new(0.0, 100.0);
//!
//! // Normalize values to [0.0, 1.0]
//! assert_eq!(scale.normalize(&0.0), 0.0);
//! assert_eq!(scale.normalize(&50.0), 0.5);
//! assert_eq!(scale.normalize(&100.0), 1.0);
//!
//! // Denormalize back to domain
//! assert_eq!(scale.denormalize(0.5), 50.0);
//! ```
//!
//! ## Pan and Zoom
//!
//! ```rust
//! use tjart::{Scale, scale::Linear};
//!
//! let mut scale = Linear::<f64, f64>::new(0.0, 100.0);
//!
//! // Pan by 10% (shifts by 10 units)
//! scale.pan(0.1);
//! assert_eq!(scale.domain(), (&10.0, &110.0));
//!
//! // Zoom in by 2x at center
//! let mut scale = Linear::<f64, f64>::new(0.0, 100.0);
//! scale.zoom(2.0, Some(0.5));
//! assert_eq!(scale.domain(), (&25.0, &75.0));
//! ```
//!
//! ## Coordinate Transformation
//!
//! ```rust
//! use tjart::{Transform, scale::Linear, ScreenRect, PlotPoint};
//!
//! let x_scale = Linear::<f64, f32>::new(0.0, 100.0);
//! let y_scale = Linear::<f64, f32>::new(0.0, 50.0);
//! let screen = ScreenRect { x: 0.0, y: 0.0, width: 800.0, height: 400.0 };
//!
//! let transform = Transform::new(&screen, &x_scale, &y_scale);
//!
//! // Convert plot coordinates to screen pixels
//! let plot_point = PlotPoint::new(50.0, 25.0);
//! let screen_point = transform.chart_to_screen(&plot_point);
//! // screen_point is at (400.0, 200.0) - center of screen
//! ```
//!
//! ## Generating Ticks
//!
//! ```rust
//! use tjart::{Scale, scale::Linear};
//!
//! let scale = Linear::<f64, f64>::new(0.0, 100.0);
//! let ticks = scale.ticks();
//!
//! // Ticks include both major (level 0) and minor (level 1) marks
//! for tick in ticks {
//!     if tick.level == 0 {
//!         println!("Major tick at: {}", tick.value);
//!     }
//! }
//! ```
//!

pub mod scale;
pub mod transform;

pub use num_traits::Float;
pub use scale::{Scale, Tick, TickIter};
pub use transform::{PlotPoint, PlotRect, ScreenPoint, ScreenRect, Transform};
