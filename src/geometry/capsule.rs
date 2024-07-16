use crate::linear::{Vec3f, Vector};

use super::{aabb::AABB, segment::Segment, shape::Shape, sphere::Sphere};

#[derive(Clone, Debug)]
pub struct Capsule<T: Vector> {
    pub start: T,
    pub end: T,
    pub radius: f32,
    pub height: f32,
}

impl<T: Vector> Capsule<T> {
    pub fn new(start: T, end: T, radius: f32) -> Self {
        let height = (end - start).magnitude();

        Self {
            start,
            end,
            radius,
            height,
        }
    }
}

impl Shape<Vec3f> for Capsule<Vec3f> {
    fn contains_point(&self, point: Vec3f) -> bool {
        todo!()
    }

    fn intersects_ray(&self, ray: &Segment<Vec3f>) -> bool {
        todo!()
    }

    fn extents(&self) -> AABB<Vec3f> {
        let sphere_a = Sphere::new(self.start, self.radius).extents();
        let sphere_b = Sphere::new(self.end, self.radius).extents();

        return AABB::new(
            Vec3f::new(
                f32::min(sphere_a.min.x, sphere_b.min.x),
                f32::min(sphere_a.min.y, sphere_b.min.y),
                f32::min(sphere_a.min.z, sphere_b.min.z),
            ),
            Vec3f::new(
                f32::max(sphere_a.max.x, sphere_b.max.x),
                f32::max(sphere_a.max.y, sphere_b.max.y),
                f32::max(sphere_a.max.z, sphere_b.max.z),
            ),
        );
    }

    fn volume(&self) -> f32 {
        let cylinder_volume = std::f32::consts::PI * self.radius * self.radius * self.height;
        let sphere_volume =
            (4.0 / 3.0) * std::f32::consts::PI * self.radius * self.radius * self.radius;

        return cylinder_volume + sphere_volume;
    }

    fn furthest_point(&self, direction: Vec3f) -> Vec3f {
        // Start at one of the circle's centre points, and add radius in the given direction.
        let centre_point = if direction.dot(self.end - self.start) >= 0.0 {
            self.end
        } else {
            self.start
        };

        return centre_point + direction.normalise() * self.radius;
    }

    fn centre(&self) -> Vec3f {
        (self.start + self.end) * 0.5
    }

    fn translate(&mut self, translation: Vec3f) {
        self.start += translation;
        self.end += translation;
    }

    fn rotate(&mut self, rotation: Vec3f) {
        todo!()
    }

    fn scale(&mut self, scale: Vec3f) {
        self.start *= scale;
        self.end *= scale;
        self.radius *= scale.x.max(scale.y).max(scale.z);
        self.height = (self.end - self.start).magnitude();
    }

    fn points(&self) -> &[Vec3f] {
        unimplemented!("A capsule does not have points");
    }
}


