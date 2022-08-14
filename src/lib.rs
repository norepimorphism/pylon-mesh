use pylon_engine::{MeshTriangle, MeshVertex, Point};

#[derive(Clone, Debug)]
pub struct Mesh {
    /// The vertices that make up this mesh.
    pub vertex_pool: Vec<MeshVertex>,
    /// Triads of vertices from [`Self::vertex_pool`] that define the triangle primitives of this
    /// mesh.
    pub triangles: Vec<MeshTriangle>
}

// STL /////////////////////////////////////////////////////////////////////////////////////////////

impl From<&stl_io::IndexedMesh> for Mesh {
    fn from(mesh: &stl_io::IndexedMesh) -> Self {
        Mesh {
            vertex_pool: {
                // Note: we could probably transmute here to guarantee a zero-cost copy, but we
                // would need to be extra careful in case the definition from `stl_io` ever changes.
                mesh
                    .vertices
                    .iter()
                    .copied()
                    .map(|v| vertex_from_stl(v))
                    .collect()
            },
            triangles: {
                mesh
                    .faces
                    .iter()
                    .map(|t| triangle_from_stl(t))
                    .collect()
            },
        }
    }
}

fn vertex_from_stl(vertex: stl_io::Vertex) -> MeshVertex {
    let [x, y, z]: [f32; 3] = vertex.into();

    MeshVertex { point: Point { x, y, z } }
}

fn triangle_from_stl(triangle: &stl_io::IndexedTriangle) -> MeshTriangle {
    MeshTriangle::new(triangle.vertices.map(|x| x as u32))
}

// Collada /////////////////////////////////////////////////////////////////////////////////////////

// impl From<&dae_parser::Mesh> for Mesh {
//     fn from(mesh: &dae_parser::Mesh) -> Self {
//         Mesh {
//             triangles: {
//                 mesh
//                     .elements
//                     .iter()
//                     .map(|e| {
//                         match e {
//                             dae_parser::Primitive::Triangles(t) => {
//                                 t.inputs.
//                                 let
//                             }
//                         }
//                     })
//             },
//         }
//     }
// }
