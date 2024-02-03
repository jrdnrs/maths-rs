use crate::linear::{Mat4f, Vec2f, Vec3f};

use super::{aabb::AABB, segment::Segment};

pub trait Shape2D {
    fn contains_point(&self, point: Vec2f) -> bool;
    fn intersects_ray(&self, ray: &Segment) -> bool;
    fn overlaps(&self, other: &impl Shape2D) -> bool {
        super::sat::separating_axis_test(self.points(), other.points())
    }
    fn extents(&self) -> AABB;
    fn area(&self) -> f32;
    fn centre(&self) -> Vec2f;
    fn translate(&mut self, translation: Vec2f);
    fn rotate(&mut self, sin: f32, cos: f32);
    fn scale(&mut self, scale: Vec2f);
    fn points(&self) -> &[Vec2f];
}

pub trait Shape3D {
    fn contains_point(&self, point: Vec3f) -> bool;
    fn intersects_ray(&self, ray: &Segment) -> bool;
    fn extents(&self) -> AABB;
    fn volume(&self) -> f32;
    fn furthest_point(&self, direction: Vec3f) -> Vec3f;
    fn centre(&self) -> Vec3f;
    fn translate(&mut self, translation: Vec3f);
    fn rotate(&mut self, rotation: Vec3f);
    fn scale(&mut self, scale: Vec3f);
    fn transform(&mut self, transform: &Mat4f);
    fn points(&self) -> &[Vec3f];
}
