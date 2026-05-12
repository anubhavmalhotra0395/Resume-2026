import { BufferGeometry } from "/three.module.min.js";

// Minimal subset required by GLTFLoader.js in this repo.
// Based on Three.js BufferGeometryUtils.toTrianglesDrawMode().
export function toTrianglesDrawMode( geometry, drawMode ) {
  if ( !( geometry && geometry.isBufferGeometry ) ) return geometry;
  if ( drawMode === 0 ) return geometry; // TrianglesDrawMode
  if ( geometry.index === null ) {
    // Non-indexed geometry: nothing we can do cheaply here.
    return geometry;
  }

  const index = geometry.getIndex();
  const numberOfTriangles = index.count - 2;
  const newIndices = [];

  if ( drawMode === 1 ) {
    // TriangleStripDrawMode
    for ( let i = 0; i < numberOfTriangles; i++ ) {
      if ( i % 2 === 0 ) {
        newIndices.push( index.getX( i ), index.getX( i + 1 ), index.getX( i + 2 ) );
      } else {
        newIndices.push( index.getX( i + 2 ), index.getX( i + 1 ), index.getX( i ) );
      }
    }
  } else if ( drawMode === 2 ) {
    // TriangleFanDrawMode
    for ( let i = 1; i <= numberOfTriangles; i++ ) {
      newIndices.push( index.getX( 0 ), index.getX( i ), index.getX( i + 1 ) );
    }
  } else {
    return geometry;
  }

  const newGeometry = geometry.clone();
  newGeometry.setIndex( newIndices );
  return newGeometry;
}

