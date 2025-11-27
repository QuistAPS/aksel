use fastnum::decimal::D128;
use tjart::{
    PlotPoint, PlotRect, Scale, ScreenPoint, ScreenRect, Transform,
    scale::{Linear, Logarithmic},
};

#[test]
fn test_linear_scale_with_decimal_domain_and_normalized() {
    // Create a linear scale with D128 (Decimal 128-bit) for both domain and normalized
    let min = D128::from(0);
    let max = D128::from(100);
    let scale = Linear::<D128, D128>::new(min, max);

    // Test normalization
    let value = D128::from(50);
    let normalized = scale.normalize(&value);
    assert!((normalized - D128::from(0.5)).abs() < D128::from(1e-10));

    // Test denormalization
    let denormalized = scale.denormalize(D128::from(0.5));
    assert!((denormalized - D128::from(50)).abs() < D128::from(1e-10));
}

#[test]
fn test_linear_scale_with_decimal_domain_f32_normalized() {
    // Create a linear scale with D128 for domain and f32 for normalized
    let min = D128::from(0);
    let max = D128::from(100);
    let scale = Linear::<D128, f32>::new(min, max);

    // Test normalization (D128 -> f32)
    let value = D128::from(50);
    let normalized: f32 = scale.normalize(&value);
    assert!((normalized - 0.5f32).abs() < 1e-6);

    // Test denormalization (f32 -> D128)
    let denormalized = scale.denormalize(0.5f32);
    assert!((denormalized - D128::from(50)).abs() < D128::from(1e-10));
}

#[test]
fn test_logarithmic_scale_with_decimal() {
    // Create a logarithmic scale with D128 for domain and normalized
    let base = D128::from(10);
    let min = D128::from(1);
    let max = D128::from(100);
    let scale = Logarithmic::<D128, D128>::new(base, min, max);

    // Test normalization at geometric midpoint (10.0)
    let value = D128::from(10);
    let normalized = scale.normalize(&value);
    // In log10 space: log10(1) = 0, log10(100) = 2, log10(10) = 1
    // Normalized should be 1/2 = 0.5
    assert!((normalized - D128::from(0.5)).abs() < D128::from(1e-10));

    // Test denormalization
    let denormalized = scale.denormalize(D128::from(0.5));
    assert!((denormalized - D128::from(10)).abs() < D128::from(1e-8));
}

#[test]
fn test_transform_with_decimal_domain_f32_screen() {
    // Create scales with D128 domain, D128 normalized, and we'll use f32 for screen
    let x_scale = Linear::<D128, D128>::new(D128::from(0), D128::from(100));
    let y_scale = Linear::<D128, D128>::new(D128::from(0), D128::from(100));

    let screen_rect = ScreenRect {
        x: 0.0f32,
        y: 0.0f32,
        width: 800.0f32,
        height: 600.0f32,
    };

    let transform = Transform::new(&screen_rect, &x_scale, &y_scale);

    // Test chart to screen conversion
    let plot_point = PlotPoint::new(D128::from(50), D128::from(50));
    let screen_point = transform.chart_to_screen(&plot_point);

    // At (50, 50) in domain, normalized is (0.5, 0.5)
    // Screen should be (400, 300) with inverted Y
    assert!((screen_point.x - 400.0f32).abs() < 1e-4);
    assert!((screen_point.y - 300.0f32).abs() < 1e-4);

    // Test screen to chart conversion
    let screen_point2 = ScreenPoint::new(400.0f32, 300.0f32);
    let plot_point2 = transform.screen_to_chart(&screen_point2);

    assert!((plot_point2.x - D128::from(50)).abs() < D128::from(1e-8));
    assert!((plot_point2.y - D128::from(50)).abs() < D128::from(1e-8));
}

#[test]
fn test_transform_mixed_decimal_and_f32() {
    // Domain: D128, Normalized: f32, Screen: f32
    let x_scale = Linear::<D128, f32>::new(D128::from(0), D128::from(100));
    let y_scale = Linear::<D128, f32>::new(D128::from(0), D128::from(100));

    let screen_rect = ScreenRect {
        x: 0.0f32,
        y: 0.0f32,
        width: 1000.0f32,
        height: 1000.0f32,
    };

    let transform = Transform::new(&screen_rect, &x_scale, &y_scale);

    // Test chart to screen
    let plot_point = PlotPoint::new(D128::from(25), D128::from(75));
    let screen_point = transform.chart_to_screen(&plot_point);

    // (25, 75) normalized to (0.25, 0.75)
    // Screen: x = 0.25 * 1000 = 250, y = (1 - 0.75) * 1000 = 250
    assert!((screen_point.x - 250.0f32).abs() < 1e-3);
    assert!((screen_point.y - 250.0f32).abs() < 1e-3);
}

#[test]
fn test_decimal_scale_operations() {
    // Test pan and zoom operations with Decimal
    let mut scale = Linear::<D128, D128>::new(D128::from(0), D128::from(100));

    // Test pan
    scale.pan(D128::from(0.1)); // Pan by 10%

    let (min, max) = scale.domain();
    assert!((*min - D128::from(10)).abs() < D128::from(1e-10));
    assert!((*max - D128::from(110)).abs() < D128::from(1e-10));

    // Test zoom
    scale = Linear::<D128, D128>::new(D128::from(0), D128::from(100));
    scale.zoom(D128::from(2), Some(D128::from(0.5))); // Zoom in 2x at center

    let (min, max) = scale.domain();
    // Should zoom to [25, 75]
    assert!((*min - D128::from(25)).abs() < D128::from(1e-10));
    assert!((*max - D128::from(75)).abs() < D128::from(1e-10));
}

#[test]
fn test_decimal_rect_transformations() {
    let x_scale = Linear::<D128, f32>::new(D128::from(0), D128::from(100));
    let y_scale = Linear::<D128, f32>::new(D128::from(0), D128::from(100));

    let screen_rect = ScreenRect {
        x: 0.0f32,
        y: 0.0f32,
        width: 1000.0f32,
        height: 1000.0f32,
    };

    let transform = Transform::new(&screen_rect, &x_scale, &y_scale);

    // Test chart rect to screen rect
    let plot_rect = PlotRect {
        x: D128::from(20),
        y: D128::from(30),
        width: D128::from(40),
        height: D128::from(20),
    };

    let screen_rect_result = transform.chart_to_screen_rect(plot_rect);

    // x: 20/100 = 0.2, width: 40/100 = 0.4
    // y: (30+20)/100 = 0.5 (top), inverted: 1 - 0.5 = 0.5
    assert!((screen_rect_result.x - 200.0f32).abs() < 1e-2);
    assert!((screen_rect_result.width - 400.0f32).abs() < 1e-2);
}
