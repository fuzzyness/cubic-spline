use plotters::prelude::*;
use std::error::Error;
use std::fs;
use std::path::Path;

/// Struct to hold spline data points and endpoint derivatives.
struct SplineData {
    node_positions: Vec<f64>,
    node_values: Vec<f64>,
    derivative_start: Option<f64>, // If Some, use clamped condition at start
    derivative_end: Option<f64>, // If Some, use clamped condition at end
}

/// Computes the second derivatives for the spline interpolation.
fn compute_second_derivative(
    node_positions: &[f64],
    node_values: &[f64],
    derivative_start: Option<f64>,
    derivative_end: Option<f64>,
) -> Vec<f64> {
    // Calculate number of intervals between nodes.
    let intervals = node_positions.len() - 1;
    // Compute the width of each interval.
    let mut interval_widths = Vec::with_capacity(intervals);
    for i in 0..intervals {
        interval_widths.push(node_positions[i + 1] - node_positions[i]);
    }

    // Total number of nodes.
    let nodes = intervals + 1;
    // Coefficients for the tridiagonal system.
    let mut a = vec![0.0; nodes];
    let mut b = vec![0.0; nodes];
    let mut c = vec![0.0; nodes];
    let mut d = vec![0.0; nodes];

    // Setup the equations for the left endpoint.
    if let Some(derivative_start) = derivative_start {
        // Clamped condition at the left endpoint.
        b[0] = 2.0 * interval_widths[0];
        c[0] = interval_widths[0];
        // d[0] contains the difference between the slope of the first segment
        // and the given derivative.
        d[0] = 6.0 * ((node_values[1] - node_values[0])
                      / interval_widths[0] - derivative_start);
    } else {
        // Natural spline: Second derivative at left endpoint equals 0.
        b[0] = 1.0;
        d[0] = 0.0;
    }

    // Setup the equations for the interior points.
    for i in 1..intervals {
        a[i] = interval_widths[i - 1];
        b[i] = 2.0 * (interval_widths[i - 1] + interval_widths[i]);
        c[i] = interval_widths[i];
        d[i] = 6.0 * ((node_values[i + 1] - node_values[i]) / interval_widths[i]
                      - (node_values[i] - node_values[i - 1]) / interval_widths[i - 1]);
    }

    // Setup the equations for the right endpoint.
    if let Some(derivative_end) = derivative_end {
        // Clamped condition at the right endpoint.
        a[intervals] = interval_widths[intervals - 1];
        b[intervals] = 2.0 * interval_widths[intervals - 1];
        d[intervals] = 6.0 * (derivative_end
                                  - (node_values[intervals] - node_values[intervals - 1])
                                  / interval_widths[intervals - 1]);
    } else {
        // Natural spline: Second derivative at right endpoint equals 0.
        b[intervals] = 1.0;
        d[intervals] = 0.0;
    }

    // Solve the tridiagonal system using the Thomas algorithm.
    // Forward elimination: modify the coefficients.
    for i in 1..nodes {
        let factor = a[i] / b[i - 1];
        b[i] -= factor * c[i - 1];
        d[i] -= factor * d[i - 1];
    }

    // Back substitution: compute the second derivatives.
    let mut second_derivatives = vec![0.0; nodes];
    second_derivatives[nodes - 1] = d[nodes - 1] / b[nodes - 1];
    for i in (0..nodes - 1).rev() {
        second_derivatives[i] = (d[i] - c[i] * second_derivatives[i + 1]) / b[i];
    }

    second_derivatives
}

/// Evaluate the cubic spline at a given evaluation point using the computed second derivatives.
fn evaluate_spline(
    node_positions: &[f64],
    node_values: &[f64],
    second_derivatives: &[f64],
    eval_point: f64,
) -> f64 {
    let nodes = node_positions.len();
    // Determine which interval contains the evaluation point.
    let mut segment_index = 0;
    while segment_index < nodes - 1 && eval_point > node_positions[segment_index + 1] {
        segment_index += 1;
    }

    // Ensure that we stay within bounds.
    if segment_index >= nodes - 1 {
        segment_index = nodes - 2;
    }

    // Calculate interval width for the selected segments.
    let interval_width = node_positions[segment_index + 1] - node_positions[segment_index];
    // Compute distance from the left node of the interval.
    let dist_into_interval = eval_point - node_positions[segment_index];
    // Compute weights for interpolation.
    let weight_a = (node_positions[segment_index + 1] - eval_point) / interval_width;
    let weight_b = dist_into_interval / interval_width;
    // Compute the spline value using the standard cubic spline formula.
    let spline_value = weight_a * node_values[segment_index]
        + weight_b * node_values[segment_index + 1]
        + ((weight_a.powi(3) - weight_a) * second_derivatives[segment_index]
           + (weight_b.powi(3) - weight_b) * second_derivatives[segment_index + 1])
        * (interval_width * interval_width) / 6.0;

    spline_value
}

/// Samples the spline over each interval with the given sampling step,
/// returning a vector of (x, S(x)) points.
fn sample_spline(
    node_positions: &[f64],
    node_values: &[f64],
    second_derivatives: &[f64],
    sampling_step: f64,
) -> Vec<(f64, f64)> {
    let mut sampled_points = Vec::new();
    let start_position = node_positions[0];
    let end_position = *node_positions.last().unwrap();
    let mut sample_position = start_position;

    // Sample the spline from the start to the end position.
    while sample_position <= end_position {
        sampled_points.push((sample_position, evaluate_spline(
            node_positions, node_values, second_derivatives, sample_position
        )));
        sample_position += sampling_step;
    }

    // Ensure the final point is included.
    if sampled_points.last().map_or(true, |&(pos, _)| (pos - end_position).abs() > 1e-6) {
        sampled_points.push((end_position, evaluate_spline(
            node_positions, node_values, second_derivatives, end_position
        )));
    }

    sampled_points
}

/// Plots the given spline curves and their original data points into an image.
fn plot_spline(
    filename: &str,
    sampled_points: &[Vec<(f64, f64)>],
    original_points: &[&[(f64, f64)]],
    caption: &str,
) -> Result<(), Box<dyn Error>> {
    // Check for the "images" directory and create it if necessary.
    let images_directory = "images";
    if !Path::new(images_directory).exists() {
        fs::create_dir_all(images_directory)?;
    }
    // Build the full file path for the image.
    let filepath = format!("{}/{}", images_directory, filename);

    // Setup the drawing area with defined x and y ranges.
    let x_range = 1.0..30.0;
    let y_range = 2.5..7.5;
    let drawing_area = BitMapBackend::new(&filepath, (1280, 720)).into_drawing_area();
    drawing_area.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&drawing_area)
        .margin(20)
        .caption(caption, ("sans-serif", 30))
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(x_range, y_range)?;
    chart.configure_mesh().draw()?;

    // Define colors for each curve.
    let colors = [&RED, &GREEN, &BLUE];
    // Draw the sampled spline curves.
    for (i, spline_curve) in sampled_points.iter().enumerate() {
        chart.draw_series(LineSeries::new(
            spline_curve.clone(),
            colors[i % colors.len()].stroke_width(2),
        ))?;
    }

    // Draw the original data points on top of the spline curves.
    for (i, points) in original_points.iter().enumerate() {
        chart.draw_series(points.iter().map(|&(x_val, y_val)| {
            Circle::new((x_val, y_val), 4, colors[i % colors.len()].filled())
        }))?;
    }
    drawing_area.present()?;
    println!("Plot saved to {}", filepath);

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    // Define spline data for the three curves.
    let spline1 = SplineData {
        node_positions: vec![1.0, 2.0, 5.0, 6.0, 7.0, 8.0, 10.0, 13.0, 17.0],
        node_values: vec![3.0, 3.7, 3.9, 4.2, 5.7, 6.6, 7.1, 6.7, 4.5],
        derivative_start: Some(1.0),
        derivative_end: Some(-0.67),
    };
    let spline2 = SplineData {
        node_positions: vec![17.0, 20.0, 23.0, 24.0, 25.0, 27.0, 27.7],
        node_values: vec![4.5, 7.0, 6.1, 5.6, 5.8, 5.2, 4.1],
        derivative_start: Some(3.0),
        derivative_end: Some(-4.0),
    };
    let spline3 = SplineData {
        node_positions: vec![27.7, 28.0, 29.0, 30.0],
        node_values: vec![4.1, 4.3, 4.1, 3.0],
        derivative_start: Some(0.33),
        derivative_end: Some(-1.5),
    };

    // Collect original data points for plotting.
    let points1: Vec<(f64, f64)> = spline1.node_positions
        .iter()
        .zip(spline1.node_values.iter())
        .map(|(&position, &value)| (position, value))
        .collect();
    let points2: Vec<(f64, f64)> = spline2.node_positions
        .iter()
        .zip(spline2.node_values.iter())
        .map(|(&position, &value)| (position, value))
        .collect();
    let points3: Vec<(f64, f64)> = spline3.node_positions
        .iter()
        .zip(spline3.node_values.iter())
        .map(|(&position, &value)| (position, value))
        .collect();
    let points = vec![
        points1.as_slice(),
        points2.as_slice(),
        points3.as_slice(),
    ];

    // Clampled Splines (Exercise 27)
    // Compute second derivatives for clamped spline conditions.
    let clamped_second_derivative1 = compute_second_derivative(
        &spline1.node_positions,
        &spline1.node_values,
        spline1.derivative_start,
        spline1.derivative_end,
    );
    let clamped_second_derivative2 = compute_second_derivative(
        &spline2.node_positions,
        &spline2.node_values,
        spline2.derivative_start,
        spline2.derivative_end,
    );
    let clamped_second_derivative3 = compute_second_derivative(
        &spline3.node_positions,
        &spline3.node_values,
        spline3.derivative_start,
        spline3.derivative_end,
    );

    // Sample each clamped spline with a sampling step for plotting.
    let clamped_curve1 = sample_spline(
        &spline1.node_positions,
        &spline1.node_values,
        &clamped_second_derivative1,
        0.1,
    );
    let clamped_curve2 = sample_spline(
        &spline2.node_positions,
        &spline2.node_values,
        &clamped_second_derivative2,
        0.1,
    );
    let clamped_curve3 = sample_spline(
        &spline3.node_positions,
        &spline3.node_values,
        &clamped_second_derivative3,
        0.1,
    );
    let clamped_curves = vec![
        clamped_curve1,
        clamped_curve2,
        clamped_curve3,
    ];

    // Plot the clamped cubic spline curves.
    plot_spline(
        "clamped.png",
        &clamped_curves,
        &points,
        "Clamped Cubic Spline",
    )?;

    // Natural Splines (Exercise 28)
    // Compute the second derivatives using natural spline conditions.
    let natural_second_derivative1 = compute_second_derivative(
        &spline1.node_positions,
        &spline1.node_values,
        None,
        None,
    );
    let natural_second_derivative2 = compute_second_derivative(
        &spline2.node_positions,
        &spline2.node_values,
        None,
        None,
    );
    let natural_second_derivative3 = compute_second_derivative(
        &spline3.node_positions,
        &spline3.node_values,
        None,
        None,
    );

    // Sample each natural spline.
    let natural_curve1 = sample_spline(
        &spline1.node_positions,
        &spline1.node_values,
        &natural_second_derivative1,
        0.1,
    );
    let natural_curve2 = sample_spline(
        &spline2.node_positions,
        &spline2.node_values,
        &natural_second_derivative2,
        0.1,
    );
    let natural_curve3 = sample_spline(
        &spline3.node_positions,
        &spline3.node_values,
        &natural_second_derivative3,
        0.1,
    );
    let natural_curves = vec![
        natural_curve1,
        natural_curve2,
        natural_curve3,
    ];

    // Plot the natural cubic spline curves.
    plot_spline(
        "natural.png",
        &natural_curves,
        &points,
        "Natural Cubic Spline",
    )?;

    Ok(())
}
