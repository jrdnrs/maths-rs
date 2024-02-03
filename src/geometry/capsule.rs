use crate::linear::{Mat4f, Vec3f};

use super::{aabb::AABB, segment::Segment, shape::Shape, sphere::Sphere};

#[derive(Clone, Debug)]
pub struct Capsule {
    pub start: Vec3f,
    pub end: Vec3f,
    pub radius: f32,
    pub height: f32,
}

impl Capsule {
    pub fn new(start: Vec3f, end: Vec3f, radius: f32) -> Self {
        let height = (end - start).magnitude();

        Self {
            start,
            end,
            radius,
            height,
        }
    }

    pub fn intersects(&self, other: &Capsule) -> bool {
        todo!()
    }

    pub fn contains(&self, other: &Capsule) -> bool {
        todo!()
    }
}

impl Shape for Capsule {
    fn contains_point(&self, point: Vec3f) -> bool {
        todo!()
    }

    fn intersects_ray(&self, ray: &Segment) -> bool {
        todo!()
    }

    fn extents(&self) -> AABB {
        let sphere_a = Sphere::new(self.start, self.radius).get_extents();
        let sphere_b = Sphere::new(self.end, self.radius).get_extents();

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


}
