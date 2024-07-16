use crate::linear::{Vec2f, Vec3f, Vector};

use super::{segment::Segment, shape::Shape};

#[derive(Debug, Clone, Copy)]
pub struct AABB<T: Vector> {
    pub min: T,
    pub max: T,
}

impl<T: Vector> AABB<T> {
    pub fn new(min: T, max: T) -> Self {
        Self { min, max }
    }

    pub fn from_dimensions(centre: T, dimensions: T) -> Self {
        let half_dimensions = dimensions * 0.5;
        Self::new(centre - half_dimensions, centre + half_dimensions)
    }
}

/*
    2D
*/

impl AABB<Vec2f> {
    pub fn intersects(&self, other: &AABB<Vec2f>) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
    }

    pub fn contains(&self, other: &AABB<Vec2f>) -> bool {
        self.min.x <= other.min.x
            && self.max.x >= other.max.x
            && self.min.y <= other.min.y
            && self.max.y >= other.max.y
    }
}

impl Shape<Vec2f> for AABB<Vec2f> {
    fn contains_point(&self, point: Vec2f) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
    }

    fn intersects_ray(&self, ray: &Segment<Vec2f>) -> bool {
        let e1 = Segment::new(
            Vec2f::new(self.min.x, self.min.y),
            Vec2f::new(self.max.x, self.min.y),
        );
        if ray.intersects_ray(&e1) {
            return true;
        }

        let e2 = Segment::new(
            Vec2f::new(self.max.x, self.min.y),
            Vec2f::new(self.max.x, self.max.y),
        );
        if ray.intersects_ray(&e2) {
            return true;
        }

        let e3 = Segment::new(
            Vec2f::new(self.max.x, self.max.y),
            Vec2f::new(self.min.x, self.max.y),
        );
        if ray.intersects_ray(&e3) {
            return true;
        }

        let e4 = Segment::new(
            Vec2f::new(self.min.x, self.max.y),
            Vec2f::new(self.min.x, self.min.y),
        );
        ray.intersects_ray(&e4)
    }

    fn extents(&self) -> AABB<Vec2f> {
        self.clone()
    }

    fn volume(&self) -> f32 {
        (self.max.x - self.min.x) * (self.max.y - self.min.y)
    }

    fn furthest_point(&self, direction: Vec2f) -> Vec2f {
        todo!()
    }

    fn centre(&self) -> Vec2f {
        (self.min + self.max) * 0.5
    }

    fn translate(&mut self, translation: Vec2f) {
        self.min += translation;
        self.max += translation;
    }

    fn scale(&mut self, scale: Vec2f) {
        self.min *= scale;
        self.max *= scale;
    }

    fn rotate(&mut self, rotation: Vec2f) {
        unimplemented!("AABB cannot be rotated")
    }

    fn points(&self) -> &[Vec2f] {
        unimplemented!("AABB does not have points")
    }
}

/*
    3D
*/

impl AABB<Vec3f> {
    pub fn intersects(&self, other: &AABB<Vec3f>) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }

    pub fn contains(&self, other: &AABB<Vec3f>) -> bool {
        self.min.x <= other.min.x
            && self.max.x >= other.max.x
            && self.min.y <= other.min.y
            && self.max.y >= other.max.y
            && self.min.z <= other.min.z
            && self.max.z >= other.max.z
    }
}

impl Shape<Vec3f> for AABB<Vec3f> {
    fn contains_point(&self, point: Vec3f) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
            && point.z >= self.min.z
            && point.z <= self.max.z
    }

    fn intersects_ray(&self, ray: &Segment<Vec3f>) -> bool {
        todo!()
    }

    fn extents(&self) -> AABB<Vec3f> {
        self.clone()
    }

    fn volume(&self) -> f32 {
        (self.max.x - self.min.x) * (self.max.y - self.min.y) * (self.max.z - self.min.z)
    }

    fn furthest_point(&self, direction: Vec3f) -> Vec3f {
        todo!()
    }

    fn centre(&self) -> Vec3f {
        (self.min + self.max) * 0.5
    }

    fn translate(&mut self, translation: Vec3f) {
        self.min += translation;
        self.max += translation;
    }

    fn scale(&mut self, scale: Vec3f) {
        self.min *= scale;
        self.max *= scale;
    }

    fn rotate(&mut self, rotation: Vec3f) {
        unimplemented!("AABB cannot be rotated")
    }

    fn points(&self) -> &[Vec3f] {
        unimplemented!("AABB does not have points")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intersects_test() {
        let aabb_1 = AABB::new(Vec2f::new(0.0, 0.0), Vec2f::new(10.0, 10.0));
        let aabb_2 = AABB::new(Vec2f::new(2.0, 2.0), Vec2f::new(8.0, 8.0));
        let aabb_3 = AABB::new(Vec2f::new(11.0, 11.0), Vec2f::new(12.0, 12.0));

        // intersects
        assert!(aabb_1.intersects(&aabb_2));

        // does not intersect
        assert!(!aabb_1.intersects(&aabb_3));
    }

    #[test]
    fn contains_test() {
        let aabb_1 = AABB::new(Vec2f::new(0.0, 0.0), Vec2f::new(10.0, 10.0));
        let aabb_2 = AABB::new(Vec2f::new(2.0, 2.0), Vec2f::new(8.0, 8.0));

        // contains
        assert!(aabb_1.contains(&aabb_2));

        // does not contain
        assert!(!aabb_2.contains(&aabb_1));
    }

    #[test]
    fn contains_point_test() {
        let aabb = AABB::new(Vec2f::new(0.0, 0.0), Vec2f::new(10.0, 10.0));

        // contains
        assert!(aabb.contains_point(Vec2f::new(2.0, 2.0)));

        // does not contain
        assert!(!aabb.contains_point(Vec2f::new(11.0, 11.0)));
    }

    #[test]
    fn area_test() {
        let aabb = AABB::new(Vec2f::new(0.0, 0.0), Vec2f::new(10.0, 10.0));

        assert_eq!(aabb.volume(), 100.0);
    }

    #[test]
    fn centre_test() {
        let aabb = AABB::new(Vec2f::new(0.0, 0.0), Vec2f::new(10.0, 10.0));

        assert_eq!(aabb.centre(), Vec2f::new(5.0, 5.0));
    }
}
