use crate::linear::Vec2f;

fn project_polygon(axis: Vec2f, vertices: &[Vec2f]) -> (f32, f32) {
    let mut min = f32::MAX;
    let mut max = f32::MIN;

    for vertex in vertices.iter() {
        let projection = vertex.dot(axis);
        min = min.min(projection);
        max = max.max(projection);
    }

    (min, max)
}

fn overlap(main_min: f32, main_max: f32, other_min: f32, other_max: f32) -> bool {
    main_max >= other_min && other_max >= main_min
}

fn contains(main_min: f32, main_max: f32, other_min: f32, other_max: f32) -> bool {
    main_min <= other_min && main_max >= other_max
}

pub fn separating_axis_test<'a>(
    axes: impl Iterator<Item = &'a Vec2f>,
    points_a: &[Vec2f],
    points_b: &[Vec2f],
) -> bool {
    for axis in axes {
        let (a_min, a_max) = project_polygon(*axis, points_a);
        let (b_min, b_max) = project_polygon(*axis, points_b);

        if !overlap(a_min, a_max, b_min, b_max) {
            return false;
        }
    }

    true
}

pub fn normal_edges(points: &[Vec2f]) -> Vec<Vec2f> {
    if points.len() == 2 {
        return vec![(points[1] - points[0]).perpendicular()];
    }

    let mut normals = Vec::with_capacity(points.len());
    let mut point = &points[points.len() - 1];

    for next_point in points.iter() {
        let edge = *next_point - *point;

        normals.push(edge.perpendicular());
        point = next_point;
    }

    normals
}

#[cfg(test)]
mod tests {
    use crate::{
        geometry::{Segment, Shape, Triangle},
        linear::Vec2f,
    };

    use super::*;

    #[test]
    fn overlaps_test() {
        let triangle = Triangle::new(
            Vec2f::new(0.0, 0.0),
            Vec2f::new(1.0, 0.0),
            Vec2f::new(0.0, 1.0),
        );

        // intersects
        let segment = Segment::new(Vec2f::new(0.2, 0.2), Vec2f::new(2.0, 2.0));
        let mut axes = normal_edges(triangle.points());
        axes.extend(normal_edges(segment.points()));
        assert!(separating_axis_test(
            axes.iter(),
            &triangle.points(),
            &segment.points()
        ));

        // does not intersect
        let segment = Segment::new(Vec2f::new(2.0, 2.0), Vec2f::new(4.0, 4.0));
        let mut axes = normal_edges(triangle.points());
        axes.extend(normal_edges(segment.points()));
        assert!(!separating_axis_test(
            axes.iter(),
            &triangle.points(),
            &segment.points()
        ));
    }
}
