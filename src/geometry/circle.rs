use std::f32::consts::PI;

use crate::linear::{Vec2f, Vector};

use super::{aabb::AABB, segment::Segment, shape::Shape};


pub struct Circle<T: Vector> {
    pub centre: T,
    pub radius: f32,
}

impl<T: Vector> Circle<T> {
    pub fn new(centre: T, radius: f32) -> Self {
        Self { centre, radius }
    }

    pub fn intersects(&self, other: &Circle<T>) -> bool {
        return (self.centre - other.centre).magnitude_sq()
            <= (self.radius + other.radius) * (self.radius + other.radius);
    }

    pub fn contains(&self, other: &Circle<T>) -> bool {
        return (self.radius * self.radius)
            >= (self.centre - other.centre).magnitude_sq() + (other.radius * other.radius);
    }
}

impl Shape<Vec2f> for Circle<Vec2f> {
    fn contains_point(&self, point: Vec2f) -> bool {
        return (point - self.centre).magnitude_sq() <= self.radius * self.radius;
    }

    fn intersects_ray(&self, ray: &Segment<Vec2f>) -> bool {
        let dist = ray.point_distance_sq(self.centre);
        return dist <= self.radius * self.radius;
    }

    fn extents(&self) -> AABB<Vec2f> {
        return AABB::new(
            self.centre - Vec2f::uniform(self.radius),
            self.centre + Vec2f::uniform(self.radius),
        );
    }

    fn volume(&self) -> f32 {
        return PI * self.radius * self.radius;
    }

    fn furthest_point(&self, direction: Vec2f) -> Vec2f {
        todo!()
    }

    fn centre(&self) -> Vec2f {
        return self.centre;
    }

    fn translate(&mut self, translation: Vec2f) {
        self.centre += translation;
    }

    fn scale(&mut self, scale: Vec2f) {
        self.centre *= scale;
        self.radius *= scale.x.max(scale.y);
    }

    fn rotate(&mut self, rotation: Vec2f) {
        let (sin, cos) = (rotation.x, rotation.y);
        self.centre = self.centre.rotate(sin, cos);
    }

    fn points(&self) -> &[Vec2f] {
        unimplemented!("A circle does not have points");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intersects_test() {
        let circle_1 = Circle::new(Vec2f::new(0.0, 0.0), 1.0);
        let circle_2 = Circle::new(Vec2f::new(0.5, 0.0), 1.0);
        let circle_3 = Circle::new(Vec2f::new(5.0, 0.0), 1.0);

        // intersects
        assert!(circle_1.intersects(&circle_2));

        // does not intersect
        assert!(!circle_1.intersects(&circle_3));
    }

    #[test]
    fn contains_test() {
        let circle_1 = Circle::new(Vec2f::new(0.0, 0.0), 1.0);
        let circle_2 = Circle::new(Vec2f::new(0.0, 0.0), 0.5);
        let circle_3 = Circle::new(Vec2f::new(0.0, 0.0), 3.0);

        // contains
        assert!(circle_1.contains(&circle_2));

        // does not contain
        assert!(!circle_1.contains(&circle_3));
    }

    #[test]
    fn contains_point_test() {
        let circle = Circle::new(Vec2f::new(0.0, 0.0), 1.0);

        // contains
        assert!(circle.contains_point(Vec2f::new(0.5, 0.5)));

        // does not contain
        assert!(!circle.contains_point(Vec2f::new(2.0, 2.0)));
    }

    #[test]
    fn intersects_ray_test() {
        let circle = Circle::new(Vec2f::new(0.0, 0.0), 1.0);

        // intersects
        assert!(circle.intersects_ray(&Segment::new(
            Vec2f::new(0.1, 0.1),
            Vec2f::new(2.0, 2.0)
        )));

        // does not intersect
        assert!(!circle.intersects_ray(&Segment::new(
            Vec2f::new(2.0, 2.0),
            Vec2f::new(3.0, 3.0)
        )));
    }
}
