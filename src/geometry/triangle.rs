use crate::linear::{Vec2f, Vec3f, Vector};

use super::{aabb::AABB, segment::Segment, shape::Shape};

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct Triangle<T: Vector> {
    pub a: T,
    pub b: T,
    pub c: T,
}

impl<T: Vector> Triangle<T> {
    pub fn new(a: T, b: T, c: T) -> Self {
        Self { a, b, c }
    }
}

impl Triangle<Vec2f> {
    /// Returns barycentric coordinates (U, V, W) of the point with respect to this triangle.
    ///
    /// This makes use of the provided `inv_para_area` that represents the inverse of the
    /// parallelogram area of the triangle, thus avoiding a division.
    pub fn barycentric_from_inv_area(&self, point: Vec2f, inv_para_area: f32) -> Vec3f {
        let bcp = Segment::new(self.c, self.b).edge_side(point);
        let cap = Segment::new(self.a, self.c).edge_side(point);
        // let abp = Segment::new(self.b, self.a).edge_side(point);

        let u = bcp * inv_para_area;
        let v = cap * inv_para_area;
        let w = 1.0 - u - v;

        return Vec3f::new(u, v, w);
    }

    /// Returns barycentric coordinates (U, V, W) of the point with respect to this triangle.
    pub fn barycentric(&self, point: Vec2f) -> Vec3f {
        let inv_para_area = 1.0 / Segment::new(self.b, self.a).edge_side(self.c);

        return self.barycentric_from_inv_area(point, inv_para_area);
    }
}

impl Shape<Vec2f> for Triangle<Vec2f> {
    fn contains_point(&self, point: Vec2f) -> bool {
        // pineda's method (same-side technique)
        let edge_1 = Segment::new(self.b, self.a).edge_side(point);
        let edge_2 = Segment::new(self.c, self.b).edge_side(point);
        let edge_3 = Segment::new(self.a, self.c).edge_side(point);

        // SAFETY: This is just transmuting to get the sign bit, it's fine.
        let sign_1 = unsafe { core::mem::transmute::<f32, u32>(edge_1) & 0x8000_0000 };
        let sign_2 = unsafe { core::mem::transmute::<f32, u32>(edge_2) & 0x8000_0000 };
        let sign_3 = unsafe { core::mem::transmute::<f32, u32>(edge_3) & 0x8000_0000 };

        sign_1 == sign_2 && sign_2 == sign_3
    }

    fn intersects_ray(&self, ray: &Segment<Vec2f>) -> bool {
        let edge_1 = Segment::new(self.b, self.a);
        if edge_1.intersects(ray) {
            return true;
        }

        let edge_2 = Segment::new(self.c, self.b);
        if edge_2.intersects(ray) {
            return true;
        }

        let edge_3 = Segment::new(self.a, self.c);
        edge_3.intersects(ray)
    }

    fn extents(&self) -> AABB<Vec2f> {
        let min_x = self.a.x.min(self.b.x).min(self.c.x);
        let min_y = self.a.y.min(self.b.y).min(self.c.y);

        let max_x = self.a.x.max(self.b.x).max(self.c.x);
        let max_y = self.a.y.max(self.b.y).max(self.c.y);

        AABB::new(Vec2f::new(min_x, min_y), Vec2f::new(max_x, max_y))
    }

    fn furthest_point(&self, direction: Vec2f) -> Vec2f {
        todo!()
    }

    fn volume(&self) -> f32 {
        let parallelogram_area = (self.b - self.a).cross(self.c - self.a).abs();
        parallelogram_area * 0.5
    }

    fn centre(&self) -> Vec2f {
        const THIRD: f32 = 1.0 / 3.0;
        let x = (self.a.x + self.b.x + self.c.x) * THIRD;
        let y = (self.a.y + self.b.y + self.c.y) * THIRD;

        Vec2f::new(x, y)
    }

    fn translate(&mut self, translation: Vec2f) {
        self.a += translation;
        self.b += translation;
        self.c += translation;
    }

    fn scale(&mut self, scale: Vec2f) {
        self.a *= scale;
        self.b *= scale;
        self.c *= scale;
    }

    fn rotate(&mut self, rotation: Vec2f) {
        let (sin, cos) = (rotation.x, rotation.y);
        self.a = self.a.rotate(sin, cos);
        self.b = self.b.rotate(sin, cos);
        self.c = self.c.rotate(sin, cos);
    }

    fn points(&self) -> &[Vec2f] {
        unsafe { std::slice::from_raw_parts(&self.a as *const _, 3) }
    }
}

impl From<[Vec2f; 3]> for Triangle<Vec2f> {
    fn from(points: [Vec2f; 3]) -> Self {
        unsafe { core::mem::transmute(points) }
    }
}

impl From<[Vec3f; 3]> for Triangle<Vec3f> {
    fn from(points: [Vec3f; 3]) -> Self {
        unsafe { core::mem::transmute(points) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contains_point() {
        let triangle = Triangle::new(
            Vec2f::new(0.0, 0.0),
            Vec2f::new(1.0, 0.0),
            Vec2f::new(0.0, 1.0),
        );

        // inside
        assert!(triangle.contains_point(Vec2f::new(0.2, 0.2)));

        // outside
        assert!(!triangle.contains_point(Vec2f::new(1.0, 1.0)));

        let triangle = Triangle::new(
            Vec2f::new(340.0, 220.0),
            Vec2f::new(360.0, 180.0),
            Vec2f::new(380.0, 220.0),
        );

        assert!(!triangle.contains_point(Vec2f::new(350.5, 160.5)));
    }

    #[test]
    fn intersects_ray_test() {
        let triangle = Triangle::new(
            Vec2f::new(0.0, 0.0),
            Vec2f::new(1.0, 0.0),
            Vec2f::new(0.0, 1.0),
        );

        // intersects
        assert!(triangle.intersects_ray(&Segment::new(Vec2f::new(0.1, 0.1), Vec2f::new(2.0, 2.0))));

        // does not intersect
        assert!(!triangle.intersects_ray(&Segment::new(Vec2f::new(2.0, 2.0), Vec2f::new(4.0, 4.0))));
    }

    #[test]
    fn test_area() {
        let triangle = Triangle::new(
            Vec2f::new(0.0, 0.0),
            Vec2f::new(1.0, 0.0),
            Vec2f::new(0.0, 1.0),
        );

        assert_eq!(triangle.volume(), 0.5);

        let triangle = Triangle::new(
            Vec2f::new(1.0, 0.0),
            Vec2f::new(2.5, 1.0),
            Vec2f::new(0.0, 4.0),
        );

        assert_eq!(triangle.volume(), 3.5);
    }
}
