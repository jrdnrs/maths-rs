use crate::linear::{Mat4f, Vec3f};

use super::{aabb::AABB, segment::Segment, shape::Shape};

/// This is specifically meant for use with triangle faces.
#[derive(Clone, Debug)]
pub struct Polyhedron {
    pub vertices: Vec<Vec3f>,
}

impl Polyhedron {
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
        }
    }

    pub fn from_vertices(vertices: Vec<Vec3f>) -> Self {
        Self { vertices }
    }

    pub fn from_cuboid(scale: Vec3f) -> Self {
        let half_size = scale * 0.5;

        let mut vertices = Vec::new();

        vertices.push(Vec3f::new(-half_size.x, -half_size.y, -half_size.z));
        vertices.push(Vec3f::new(-half_size.x, -half_size.y, half_size.z));
        vertices.push(Vec3f::new(-half_size.x, half_size.y, -half_size.z));
        vertices.push(Vec3f::new(-half_size.x, half_size.y, half_size.z));
        vertices.push(Vec3f::new(half_size.x, -half_size.y, -half_size.z));
        vertices.push(Vec3f::new(half_size.x, -half_size.y, half_size.z));
        vertices.push(Vec3f::new(half_size.x, half_size.y, -half_size.z));
        vertices.push(Vec3f::new(half_size.x, half_size.y, half_size.z));

        return Self::from_vertices(vertices);
    }
}

impl Shape for Polyhedron {
    fn contains_point(&self, point: Vec3f) -> bool {
        // let ray = Segment::new(point, Vec3f::new(0.0, 0.0, f32::MAX));

        // let mut inside = false;

        // for i in (0..self.vertices.len()).step_by(3) {
        //     if ray.intersects_triangle(self.vertices[i], self.vertices[i + 1], self.vertices[i + 2])
        //     {
        //         inside = !inside;
        //     }
        // }

        // return inside;
        todo!()
    }

    fn intersects_ray(&self, ray: &Segment) -> bool {
        // for i in (0..self.vertices.len()).step_by(3) {
        //     if ray.intersects_triangle(self.vertices[i], self.vertices[i + 1], self.vertices[i + 2])
        //     {
        //         return true;
        //     }
        // }

        // return false;
        todo!()
    }

    fn extents(&self) -> AABB {
        let mut min_extent = Vec3f::uniform(f32::MAX);
        let mut max_extent = Vec3f::uniform(f32::MIN);

        for vertex in self.vertices.iter() {
            min_extent.x = f32::min(min_extent.x, vertex.x);
            min_extent.y = f32::min(min_extent.y, vertex.y);
            min_extent.z = f32::min(min_extent.z, vertex.z);

            max_extent.x = f32::max(max_extent.x, vertex.x);
            max_extent.y = f32::max(max_extent.y, vertex.y);
            max_extent.z = f32::max(max_extent.z, vertex.z);
        }

        return AABB::new(min_extent, max_extent);
    }

    fn volume(&self) -> f32 {
        todo!()
    }

    fn furthest_point(&self, direction: Vec3f) -> Vec3f {
        let mut furthest_point = &self.vertices[0];
        let mut furthest_distance = f32::MIN;

        for vertex in self.vertices.iter() {
            let projection = vertex.dot(direction);

            if projection > furthest_distance {
                furthest_distance = projection;
                furthest_point = vertex
            }
        }

        return furthest_point.clone();
    }

    fn centre(&self) -> Vec3f {
        let mut centre = Vec3f::new(0.0, 0.0, 0.0);

        for vertex in self.vertices.iter() {
            centre += *vertex;
        }

        return centre / self.vertices.len() as f32;
    }
}
