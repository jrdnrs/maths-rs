use crate::linear::Vector;

use super::{aabb::AABB, segment::Segment};

pub trait Shape<T: Vector> {
    fn contains_point(&self, point: T) -> bool;
    fn intersects_ray(&self, ray: &Segment<T>) -> bool;
    fn extents(&self) -> AABB<T>;
    fn volume(&self) -> f32;
    fn furthest_point(&self, direction: T) -> T;
    fn centre(&self) -> T;
    fn translate(&mut self, translation: T);
    fn rotate(&mut self, rotation: T);
    fn scale(&mut self, scale: T);
    fn points(&self) -> &[T];
}
