use crate::linear::{Vec2f, Vec3f, Vector};

use super::{aabb::AABB, poly::Poly, shape::Shape};

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Segment<T: Vector> {
    pub a: T,
    pub b: T,
}

impl<T: Vector> Segment<T> {
    pub const ZERO: Self = Self {
        a: T::ZERO,
        b: T::ZERO,
    };

    pub fn new(a: T, b: T) -> Self {
        Self { a, b }
    }

    pub fn length_sq(&self) -> f32 {
        (self.b - self.a).magnitude_sq()
    }

    pub fn length(&self) -> f32 {
        (self.b - self.a).magnitude()
    }

    pub fn direction(&self) -> T {
        (self.b - self.a).normalise()
    }
}

/*
    2D
*/

impl Segment<Vec2f> {
    /// Effectively returns double the signed area (parallelogram) of the triangle formed by the
    /// segment and the point. The sign of the result indicates the side of the segment the point
    /// is on (winding order).
    ///
    /// A positive result indicates a clockwise winding order, and a negative result indicates an
    /// anti-clockwise winding order.
    pub fn edge_side(&self, point: Vec2f) -> f32 {
        let edge = self.b - self.a;
        let point = point - self.a;

        return point.cross(edge);
    }

    /// Returns the intersection point of two segments, if they intersect.
    pub fn intersection(&self, other: &Segment<Vec2f>) -> Option<Vec2f> {
        let edge_1 = self.b - self.a;
        let edge_2 = other.b - other.a;

        let cross = edge_1.cross(edge_2);

        if cross == 0.0 {
            return None;
        }

        let start = other.a - self.a;
        let denom = 1.0 / cross;

        let t = start.cross(edge_2) * denom;
        let u = start.cross(edge_1) * denom;

        if t >= 0.0 && t <= 1.0 && u >= 0.0 && u <= 1.0 {
            return Some(self.a + edge_1 * t);
        }

        return None;
    }

    /// Returns whether two segments intersect.
    pub fn intersects(&self, other: &Segment<Vec2f>) -> bool {
        let edge_1 = self.b - self.a;
        let edge_2 = other.b - other.a;

        let cross = edge_1.cross(edge_2);

        // If the segments are parallel, they cannot intersect
        if cross == 0.0 {
            return false;
        }

        let start = other.a - self.a;
        let denom = 1.0 / cross;

        let t = start.cross(edge_2) * denom;
        let u = start.cross(edge_1) * denom;

        return t >= 0.0 && t <= 1.0 && u >= 0.0 && u <= 1.0;
    }

    /// Returns whether the segment overlaps with the area of an axis-aligned bounding box.
    pub fn overlaps_bounds(&self, bounds: &AABB<Vec2f>) -> bool {
        // If either point is inside the bounds, then it intersects or is inside completely
        if bounds.contains_point(self.a) || bounds.contains_point(self.b) {
            return true;
        }

        // If both points are outside the bounds on the same side, then the segment cannot intersect
        if self.a.x < bounds.min.x && self.b.x < bounds.min.x {
            return false;
        }
        if self.a.x > bounds.max.x && self.b.x > bounds.max.x {
            return false;
        }
        if self.a.y < bounds.min.y && self.b.y < bounds.min.y {
            return false;
        }
        if self.a.y > bounds.max.y && self.b.y > bounds.max.y {
            return false;
        }

        // Both points are outside but we cannot easily tell if it intersects, so we resort to
        // more expensive checks
        if bounds.intersects_ray(self) {
            return true;
        }

        // Otherwise it is outside the bounds completely
        return false;
    }

    /// Returns whether the segment overlaps with the area of a polygon.
    pub fn overlaps_polygon(&self, polygon: &Poly<Vec2f>) -> bool {
        // If both points are outside the bounds on the same side, then the segment cannot intersect.
        // This acts as an early out for the more expensive checks (not sure of the performance gain though)
        let bounds = polygon.extents();
        if self.a.x < bounds.min.x && self.b.x < bounds.min.x {
            return false;
        }
        if self.a.x > bounds.max.x && self.b.x > bounds.max.x {
            return false;
        }
        if self.a.y < bounds.min.y && self.b.y < bounds.min.y {
            return false;
        }
        if self.a.y > bounds.max.y && self.b.y > bounds.max.y {
            return false;
        }

        // If it intersects the polygon, then it overlaps
        if polygon.intersects_ray(self) {
            return true;
        }

        // If either point is inside the polygon, then it intersects or is inside completely
        if polygon.contains_point(self.a) || polygon.contains_point(self.b) {
            return true;
        }

        // Otherwise it is outside the polygon completely
        return false;
    }

    /// Clips a segment to an axis-aligned bounding box
    pub fn clip_bounds(&self, bounds: &AABB<Vec2f>) -> Segment<Vec2f> {
        let mut points = [self.a, self.b];

        // If the segment is entirely to one side of the bounds, it cannot intersect, so we
        // can just clip it to the bounds directly (like projecting it onto the bounds)
        if self.a.x < bounds.min.x && self.b.x < bounds.min.x {
            points[0].x = bounds.min.x;
            points[1].x = bounds.min.x;
            points[0].y = points[0].y.max(bounds.min.y).min(bounds.max.y);
            points[1].y = points[1].y.max(bounds.min.y).min(bounds.max.y);

            return Segment::new(points[0], points[1]);
        }
        if self.a.x > bounds.max.x && self.b.x > bounds.max.x {
            points[0].x = bounds.max.x;
            points[1].x = bounds.max.x;
            points[0].y = points[0].y.max(bounds.min.y).min(bounds.max.y);
            points[1].y = points[1].y.max(bounds.min.y).min(bounds.max.y);

            return Segment::new(points[0], points[1]);
        }
        if self.a.y < bounds.min.y && self.b.y < bounds.min.y {
            points[0].x = points[0].x.max(bounds.min.x).min(bounds.max.x);
            points[1].x = points[1].x.max(bounds.min.x).min(bounds.max.x);
            points[0].y = bounds.min.y;
            points[1].y = bounds.min.y;

            return Segment::new(points[0], points[1]);
        }
        if self.a.y > bounds.max.y && self.b.y > bounds.max.y {
            points[0].x = points[0].x.max(bounds.min.x).min(bounds.max.x);
            points[1].x = points[1].x.max(bounds.min.x).min(bounds.max.x);
            points[0].y = bounds.max.y;
            points[1].y = bounds.max.y;

            return Segment::new(points[0], points[1]);
        }

        let y1 = self.a.y;
        let x1 = self.a.x;
        let x2 = self.b.x;
        let y2 = self.b.y;

        for point in points.iter_mut() {
            if point.x < bounds.min.x {
                point.x = bounds.min.x;
                point.y = y1 + (y2 - y1) / (x2 - x1) * (bounds.min.x - x1);
            } else if point.x > bounds.max.x {
                point.x = bounds.max.x;
                point.y = y1 + (y2 - y1) / (x2 - x1) * (bounds.max.x - x1);
            }

            if point.y < bounds.min.y {
                point.x = x1 + (x2 - x1) / (y2 - y1) * (bounds.min.y - y1);
                point.y = bounds.min.y;
            } else if point.y > bounds.max.y {
                point.x = x1 + (x2 - x1) / (y2 - y1) * (bounds.max.y - y1);
                point.y = bounds.max.y;
            }
        }

        return Segment::new(points[0], points[1]);
    }

    /// Returns squared shortest distance from the segment to the point.
    pub fn point_distance_sq(&self, point: Vec2f) -> f32 {
        let ab = self.b - self.a;
        let ap = point - self.a;

        let displacement_proj = ap.dot(ab);
        let ab_sq = ab.dot(ab);

        assert!(ab_sq > 0.0, "Segment length is zero");
        let t = displacement_proj / ab_sq;

        if t >= 0.0 && t <= 1.0 {
            let projection = self.a + ab * t;
            return (projection - point).magnitude_sq();
        } else {
            let dist_a = (point - self.a).magnitude_sq();
            let dist_b = (point - self.b).magnitude_sq();

            return dist_a.min(dist_b);
        }
    }

    /// Returns shortest distance from the segment to the point
    pub fn point_distance(&self, point: Vec2f) -> f32 {
        return self.point_distance_sq(point).sqrt();
    }

    pub fn normal(&self) -> Vec2f {
        let dir = self.direction();
        return Vec2f::new(-dir.y, dir.x);
    }
}

impl Shape<Vec2f> for Segment<Vec2f> {
    fn contains_point(&self, point: Vec2f) -> bool {
        let ab = self.b - self.a;
        let ap = point - self.a;

        let t = ap.dot(ab) / ab.dot(ab);

        return t >= 0.0 && t <= 1.0;
    }

    fn intersects_ray(&self, ray: &Segment<Vec2f>) -> bool {
        self.intersects(ray)
    }

    fn extents(&self) -> AABB<Vec2f> {
        let min_x = self.a.x.min(self.b.x);
        let min_y = self.a.y.min(self.b.y);

        let max_x = self.a.x.max(self.b.x);
        let max_y = self.a.y.max(self.b.y);

        return AABB::new(Vec2f::new(min_x, min_y), Vec2f::new(max_x, max_y));
    }

    fn volume(&self) -> f32 {
        return 0.0;
    }

    fn furthest_point(&self, direction: Vec2f) -> Vec2f {
        todo!()
    }

    fn centre(&self) -> Vec2f {
        return (self.a + self.b) * 0.5;
    }

    fn translate(&mut self, translation: Vec2f) {
        self.a += translation;
        self.b += translation;
    }

    fn scale(&mut self, scale: Vec2f) {
        self.a *= scale;
        self.b *= scale;
    }

    fn rotate(&mut self, rotation: Vec2f) {
        let (sin, cos) = (rotation.x, rotation.y);
        self.a = self.a.rotate(sin, cos);
        self.b = self.b.rotate(sin, cos);
    }

    fn points(&self) -> &[Vec2f] {
        return unsafe { std::slice::from_raw_parts(&self.a as *const _, 2) };
    }
}

/*
    3D
*/

impl Shape<Vec3f> for Segment<Vec3f> {
    fn contains_point(&self, point: Vec3f) -> bool {
        let ab = self.b - self.a;
        let ap = point - self.a;

        let t = ap.dot(ab) / ab.dot(ab);

        return t >= 0.0 && t <= 1.0;
    }

    fn intersects_ray(&self, ray: &Segment<Vec3f>) -> bool {
        todo!()
    }

    fn extents(&self) -> AABB<Vec3f> {
        let min_x = self.a.x.min(self.b.x);
        let min_y = self.a.y.min(self.b.y);
        let min_z = self.a.z.min(self.b.z);

        let max_x = self.a.x.max(self.b.x);
        let max_y = self.a.y.max(self.b.y);
        let max_z = self.a.z.max(self.b.z);

        return AABB::new(
            Vec3f::new(min_x, min_y, min_z),
            Vec3f::new(max_x, max_y, max_z),
        );
    }

    fn volume(&self) -> f32 {
        return 0.0;
    }

    fn furthest_point(&self, direction: Vec3f) -> Vec3f {
        todo!()
    }

    fn centre(&self) -> Vec3f {
        return (self.a + self.b) * 0.5;
    }

    fn translate(&mut self, translation: Vec3f) {
        self.a += translation;
        self.b += translation;
    }

    fn scale(&mut self, scale: Vec3f) {
        self.a *= scale;
        self.b *= scale;
    }

    fn rotate(&mut self, rotation: Vec3f) {
        todo!()
    }

    fn points(&self) -> &[Vec3f] {
        return unsafe { std::slice::from_raw_parts(&self.a as *const _, 2) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intersects_test() {
        let seg_1 = Segment::new(Vec2f::new(0.0, 0.0), Vec2f::new(10.0, 10.0));
        let seg_2 = Segment::new(Vec2f::new(0.0, 5.0), Vec2f::new(10.0, 0.0));
        let seg_3 = Segment::new(Vec2f::new(10.0, 0.0), Vec2f::new(20.0, 10.0));
        let seg_4 = Segment::new(Vec2f::new(20.0, 0.0), Vec2f::new(25.0, 30.0));

        // intersects
        assert!(seg_1.intersects(&seg_2));

        // does not intersect (parallel)
        assert!(!seg_1.intersects(&seg_3));

        // does not intersect (not parallel)
        assert!(!seg_1.intersects(&seg_4));
    }

    #[test]
    fn intersection_test() {
        let seg_1 = Segment::new(Vec2f::new(0.0, 0.0), Vec2f::new(10.0, 10.0));
        let seg_2 = Segment::new(Vec2f::new(0.0, 5.0), Vec2f::new(10.0, 0.0));
        let seg_3 = Segment::new(Vec2f::new(10.0, 0.0), Vec2f::new(20.0, 10.0));
        let seg_4 = Segment::new(Vec2f::new(20.0, 0.0), Vec2f::new(25.0, 30.0));

        // intersects
        let intersection = seg_1.intersection(&seg_2).unwrap();
        let answer = Vec2f::new(10.0 / 3.0, 10.0 / 3.0);
        let delta = (intersection - answer).abs();
        assert!(delta.x < 0.0001 && delta.y < 0.0001);

        // does not intersect (parallel)
        assert_eq!(seg_1.intersection(&seg_3), None);

        // does not intersect (not parallel)
        assert_eq!(seg_1.intersection(&seg_4), None);
    }

    #[test]
    fn length_sq_test() {
        let seg = Segment::new(Vec2f::new(0.0, 0.0), Vec2f::new(10.0, 10.0));

        assert_eq!(seg.length_sq(), 200.0);
    }

    #[test]
    fn length_test() {
        let seg = Segment::new(Vec2f::new(0.0, 0.0), Vec2f::new(10.0, 10.0));

        assert_eq!(seg.length(), f32::sqrt(200.0));
    }

    #[test]
    fn direction_test() {
        let seg = Segment::new(Vec2f::new(0.0, 0.0), Vec2f::new(10.0, 10.0));

        let dir = seg.direction();
        let answer = Vec2f::new(1.0 / f32::sqrt(2.0), 1.0 / f32::sqrt(2.0));
        let delta = (dir - answer).abs();
        assert!(delta.x < 0.0001 && delta.y < 0.0001);
    }

    #[test]
    fn normal_test() {
        let seg = Segment::new(Vec2f::new(0.0, 0.0), Vec2f::new(10.0, 10.0));

        let normal = seg.normal();
        let answer = Vec2f::new(-1.0 / f32::sqrt(2.0), 1.0 / f32::sqrt(2.0));
        let delta = (normal - answer).abs();
        assert!(delta.x < 0.0001 && delta.y < 0.0001);
    }

    /// One point is inside the bounds
    #[test]
    fn clip_bounds_test_half_in() {
        let seg = Segment::new(Vec2f::new(6.0, 4.0), Vec2f::new(13.0, 9.0));
        let bounds = AABB::new(Vec2f::ZERO, Vec2f::new(10.0, 10.0));

        let clipped = seg.clip_bounds(&bounds);

        let answer = Segment::new(Vec2f::new(6.0, 4.0), Vec2f::new(10.0, 6.857143));

        assert_eq!(clipped.a, answer.a, "Start point is incorrect");
        assert_eq!(clipped.b, answer.b, "End point is incorrect");
    }

    /// Both points are inside the bounds
    #[test]
    fn clip_bounds_test_all_in() {
        let seg = Segment::new(Vec2f::new(6.0, 4.0), Vec2f::new(8.0, 9.0));
        let bounds = AABB::new(Vec2f::ZERO, Vec2f::new(10.0, 10.0));

        let clipped = seg.clip_bounds(&bounds);

        let answer = Segment::new(Vec2f::new(6.0, 4.0), Vec2f::new(8.0, 9.0));

        assert_eq!(clipped.a, answer.a, "Start point is incorrect");
        assert_eq!(clipped.b, answer.b, "End point is incorrect");
    }

    /// Both points are outside the bounds, and no intersection
    #[test]
    fn clip_bounds_test_all_out() {
        let seg = Segment::new(Vec2f::new(16.0, 4.0), Vec2f::new(18.0, 19.0));
        let bounds = AABB::new(Vec2f::ZERO, Vec2f::new(10.0, 10.0));

        let clipped = seg.clip_bounds(&bounds);

        let answer = Segment::new(Vec2f::new(10.0, 4.0), Vec2f::new(10.0, 10.0));

        assert_eq!(clipped.a, answer.a, "Start point is incorrect");
        assert_eq!(clipped.b, answer.b, "End point is incorrect");
    }

    /// Both points are outside the bounds, but the segment intersects the bounds
    #[test]
    fn clip_bounds_test_all_out_intersects() {
        let seg = Segment::new(Vec2f::new(7.0, 11.0), Vec2f::new(13.0, 7.0));
        let bounds = AABB::new(Vec2f::ZERO, Vec2f::new(10.0, 10.0));

        let clipped = seg.clip_bounds(&bounds);

        let answer = Segment::new(Vec2f::new(8.5, 10.0), Vec2f::new(10.0, 9.0));

        assert_eq!(clipped.a, answer.a, "Start point is incorrect");
        assert_eq!(clipped.b, answer.b, "End point is incorrect");
    }

    #[test]
    fn overlaps_bounds_test() {
        let bounds = AABB::new(Vec2f::new(0.0, 0.0), Vec2f::new(10.0, 10.0));

        // overlaps (inside completely)
        assert_eq!(
            Segment::new(Vec2f::new(1.0, 1.0), Vec2f::new(5.0, 5.0)).overlaps_bounds(&bounds),
            true
        );

        // overlaps (intersects)
        assert_eq!(
            Segment::new(Vec2f::new(5.0, 5.0), Vec2f::new(15.0, 15.0)).overlaps_bounds(&bounds),
            true
        );

        // does not overlap (outside on same side)
        assert_eq!(
            Segment::new(Vec2f::new(-1.0, -1.0), Vec2f::new(-1.0, -5.0)).overlaps_bounds(&bounds),
            false
        );

        // does not overlap (outside on opposite sides)
        assert_eq!(
            Segment::new(Vec2f::new(-1.0, 15.0), Vec2f::new(11.0, 15.0)).overlaps_bounds(&bounds),
            false
        );
    }

    #[test]
    fn overlaps_polygon_test() {
        let polygon = Poly::from_vertices(vec![
            Vec2f::new(1.0, 3.0),
            Vec2f::new(3.0, 3.0),
            Vec2f::new(4.0, 1.0),
            Vec2f::new(1.0, 1.0),
        ]);

        // overlaps (inside completely)
        assert_eq!(
            Segment::new(Vec2f::new(1.5, 1.5), Vec2f::new(2.5, 2.5)).overlaps_polygon(&polygon),
            true
        );

        // overlaps (intersects, half inside)
        assert_eq!(
            Segment::new(Vec2f::new(2.0, 2.0), Vec2f::new(5.0, 2.0)).overlaps_polygon(&polygon),
            true
        );

        // overlaps (intersects, both points outside)
        assert_eq!(
            Segment::new(Vec2f::new(-1.0, 2.0), Vec2f::new(4.0, 2.0)).overlaps_polygon(&polygon),
            true
        );

        // does not overlap (outside on same side)
        assert_eq!(
            Segment::new(Vec2f::new(-1.0, 1.5), Vec2f::new(-1.0, 2.5)).overlaps_polygon(&polygon),
            false
        );

        // does not overlap (outside on opposite sides)
        assert_eq!(
            Segment::new(Vec2f::new(-2.0, -2.0), Vec2f::new(2.0, 6.0)).overlaps_polygon(&polygon),
            false
        );
    }
}
