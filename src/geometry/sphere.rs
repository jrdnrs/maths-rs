use crate::linear::{Mat4f, Vec3f, Vec4f};

use super::{aabb::AABB, segment::Segment, shape::Shape};

#[derive(Clone, Debug)]
pub struct Sphere {
    pub centre: Vec3f,
    pub radius: f32,
}

impl Sphere {
    pub fn new(centre: Vec3f, radius: f32) -> Self {
        Self { centre, radius }
    }

    pub fn intersects(&self, other: &Sphere) -> bool {
        let displacement = self.centre - other.centre;
        let distance_sq = displacement.dot(displacement);

        distance_sq <= (self.radius + other.radius) * (self.radius + other.radius)
    }

    pub fn contains(&self, other: &Sphere) -> bool {
        let displacement = self.centre - other.centre;
        let distance_sq = displacement.dot(displacement);

        (self.radius * self.radius) >= distance_sq + (other.radius * other.radius)
    }
}

impl Shape for Sphere {
    fn contains_point(&self, point: Vec3f) -> bool {
        let displacement = self.centre - point;
        let distance_sq = displacement.dot(displacement);

        distance_sq <= self.radius * self.radius
    }

    fn intersects_ray(&self, ray: &Segment) -> bool {
        todo!()
    }

    fn extents(&self) -> AABB {
        AABB::new(
            self.centre - Vec3f::uniform(self.radius),
            self.centre + Vec3f::uniform(self.radius),
        )
    }

    fn volume(&self) -> f32 {
        (4.0 / 3.0) * std::f32::consts::PI * self.radius * self.radius * self.radius
    }

    fn furthest_point(&self, direction: Vec3f) -> Vec3f {
        self.centre + direction.normalise() * self.radius
    }

    fn centre(&self) -> Vec3f {
        self.centre
    }
}

mod tests {
    use super::*;

    #[test]
    fn contains_test() {
        let sphere = Sphere::new(Vec3f::new(0.0, 0.0, 0.0), 1.0);
        let other = Sphere::new(Vec3f::new(0.3, 0.3, 0.0), 0.5);

        assert!(sphere.contains(&other));

        let other = Sphere::new(Vec3f::new(0.3, 0.3, 0.0), 1.5);

        assert!(!sphere.contains(&other));
    }

    #[test]
    fn intersects_test() {
        let sphere = Sphere::new(Vec3f::new(0.0, 0.0, 0.0), 1.0);
        let other = Sphere::new(Vec3f::new(1.0, 0.0, 0.0), 1.0);

        assert!(sphere.intersects(&other));

        let other = Sphere::new(Vec3f::new(5.0, 5.0, 0.0), 1.0);

        assert!(!sphere.intersects(&other));
    }
}
