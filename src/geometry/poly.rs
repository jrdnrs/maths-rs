use crate::linear::{Vec2f, Vec3f, Vector};

use super::{aabb::AABB, segment::Segment, shape::Shape};

pub struct Poly<T: Vector> {
    pub vertices: Vec<T>,
}

impl<T: Vector> Poly<T> {
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
        }
    }

    pub fn from_vertices(vertices: Vec<T>) -> Self {
        Self { vertices }
    }
}

impl Shape<Vec2f> for Poly<Vec2f> {
    fn contains_point(&self, point: Vec2f) -> bool {
        let ray = Segment::new(point, Vec2f::new(f32::MAX, point.y));

        let mut inside = false;

        for i in 0..self.vertices.len() {
            let j = (i + 1) % self.vertices.len();

            let edge = Segment::new(self.vertices[i], self.vertices[j]);

            if edge.intersects(&ray) {
                inside = !inside;
            }
        }

        return inside;
    }

    fn intersects_ray(&self, ray: &Segment<Vec2f>) -> bool {
        for i in 0..self.vertices.len() {
            let j = (i + 1) % self.vertices.len();

            let edge = Segment::new(self.vertices[i], self.vertices[j]);

            if edge.intersects(ray) {
                return true;
            }
        }

        return false;
    }

    fn extents(&self) -> AABB<Vec2f> {
        let mut min_x = f32::MAX;
        let mut min_y = f32::MAX;

        let mut max_x = f32::MIN;
        let mut max_y = f32::MIN;

        for vertex in self.vertices.iter() {
            min_x = min_x.min(vertex.x);
            min_y = min_y.min(vertex.y);

            max_x = max_x.max(vertex.x);
            max_y = max_y.max(vertex.y);
        }

        return AABB::new(Vec2f::new(min_x, min_y), Vec2f::new(max_x, max_y));
    }

    fn volume(&self) -> f32 {
        let mut area = 0.0;

        for i in 0..self.vertices.len() {
            let j = (i + 1) % self.vertices.len();

            area += self.vertices[i].cross(self.vertices[j]);
        }

        return area.abs() * 0.5;
    }

    fn furthest_point(&self, direction: Vec2f) -> Vec2f {
        todo!()
    }

    fn centre(&self) -> Vec2f {
        let mut centre = Vec2f::ZERO;

        for vertex in self.vertices.iter() {
            centre += *vertex;
        }

        return centre / self.vertices.len() as f32;
    }

    fn translate(&mut self, translation: Vec2f) {
        for vertex in self.vertices.iter_mut() {
            *vertex += translation;
        }
    }

    fn scale(&mut self, scale: Vec2f) {
        for vertex in self.vertices.iter_mut() {
            *vertex *= scale;
        }
    }

    fn rotate(&mut self, rotation: Vec2f) {
        let (sin, cos) = (rotation.x, rotation.y);
        for vertex in self.vertices.iter_mut() {
            *vertex = vertex.rotate(sin, cos);
        }
    }

    fn points(&self) -> &[Vec2f] {
        return &self.vertices;
    }
}

impl Shape<Vec3f> for Poly<Vec3f> {
    fn contains_point(&self, point: Vec3f) -> bool {
        todo!()
    }

    fn intersects_ray(&self, ray: &Segment<Vec3f>) -> bool {
        todo!()
    }

    fn extents(&self) -> AABB<Vec3f> {
        let mut min_x = f32::MAX;
        let mut min_y = f32::MAX;
        let mut min_z = f32::MAX;

        let mut max_x = f32::MIN;
        let mut max_y = f32::MIN;
        let mut max_z = f32::MIN;

        for vertex in self.vertices.iter() {
            min_x = min_x.min(vertex.x);
            min_y = min_y.min(vertex.y);
            min_z = min_z.min(vertex.z);

            max_x = max_x.max(vertex.x);
            max_y = max_y.max(vertex.y);
            max_z = max_z.max(vertex.z);
        }

        return AABB::new(
            Vec3f::new(min_x, min_y, min_z),
            Vec3f::new(max_x, max_y, max_z),
        );
    }

    fn volume(&self) -> f32 {
        todo!()
    }

    fn furthest_point(&self, direction: Vec3f) -> Vec3f {
        todo!()
    }

    fn centre(&self) -> Vec3f {
        let mut centre = Vec3f::ZERO;

        for vertex in self.vertices.iter() {
            centre += *vertex;
        }

        return centre / self.vertices.len() as f32;
    }

    fn translate(&mut self, translation: Vec3f) {
        for vertex in self.vertices.iter_mut() {
            *vertex += translation;
        }
    }

    fn scale(&mut self, scale: Vec3f) {
        for vertex in self.vertices.iter_mut() {
            *vertex *= scale;
        }
    }

    fn rotate(&mut self, rotation: Vec3f) {
        todo!()
    }

    fn points(&self) -> &[Vec3f] {
        return &self.vertices;
    }
}
