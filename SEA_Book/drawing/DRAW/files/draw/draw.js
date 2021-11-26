/**
 * Created by MURAT on 6.5.2017.
 */
if (!Detector.webgl ) Detector.addGetWebGLMessage();

var stats;

var camera, controls, scene, renderer;

var lut, legendLayout, colorMap, numberOfColors, scene2, camera2;

init();
render(); // remove when using next line for animation loop (requestAnimationFrame)
//animate();

function fitCameraToSelection( camera, controls, selection, fitOffset = 1.2 ) {

  const box = new THREE.Box3();

  for( const object of selection ) box.expandByObject( object );

  const size = box.getSize( new THREE.Vector3() );
  const center = box.getCenter( new THREE.Vector3() );

  const maxSize = Math.max( size.x, size.y, size.z );
  const fitHeightDistance = maxSize / ( 2 * Math.atan( Math.PI * camera.fov / 360 ) );
  const fitWidthDistance = fitHeightDistance / camera.aspect;
  const distance = fitOffset * Math.max( fitHeightDistance, fitWidthDistance );

  const direction = controls.target.clone()
    .sub( camera.position )
    .normalize()
    .multiplyScalar( distance );

  controls.maxDistance = distance * 10;
  controls.target.copy( center );

  camera.near = distance / 100;
  camera.far = distance * 100;
  camera.updateProjectionMatrix();

  camera.position.copy( controls.target ).sub(direction);

  controls.update();

}


function init() {

    scene = new THREE.Scene();
    scene2 = new THREE.Scene();
    //scene.fog = new THREE.FogExp2( 0xcccccc, 0.002 );

    renderer = new THREE.WebGLRenderer({antialias:true});
    renderer.autoClear = false;


    colorMap = 'rainbow';
    numberOfColors = 256;
    legendLayout = 'horizontal';
    loadModel( colorMap, numberOfColors, legendLayout );

    //renderer.setClearColor(  0xDCDCDC, 1 );
    renderer.setClearColor(  0xFFFFFF, 1 );
    renderer.setPixelRatio( window.devicePixelRatio );
    renderer.setSize( window.innerWidth, window.innerHeight );

    var container = document.getElementById( 'container' );
    container.appendChild( renderer.domElement );

    camera = new THREE.PerspectiveCamera( 10, window.innerWidth / window.innerHeight, 1, 10000000 );
    camera.position.z = 100;

    // var width = window.innerWidth;
    // var height = window.innerHeight;
    // camera2 = new THREE.OrthographicCamera( width / - 2, width / 2, height / 2, height / - 2, 1, 1000000 );
    // camera2.position.x = 0;
    // camera2.position.y = 0;
    // camera2.position.z = 100;
    camera2 = new THREE.PerspectiveCamera( 10, window.innerWidth / window.innerHeight, 1, 10000000 );
    camera2.position.x = 0;
    camera2.position.y = 3.5;
    camera2.position.z = 50;

    controls = new THREE.OrbitControls( camera, renderer.domElement );
    controls.addEventListener( 'change', render ); // remove when using animation loop

    // enable animation loop when using damping or autorotation
    controls.enableDamping = true;
    controls.dampingFactor = 0.25;
    controls.enableZoom = true;
    controls.enablePan = true;


    var geometry = new THREE.Geometry();
    cell.forEach(function(c, i){
            //console.log(c, i);
            var v1 = new THREE.Vector3(xyz[c[0]][0],xyz[c[0]][1],xyz[c[0]][2]);
            var v2 = new THREE.Vector3(xyz[c[1]][0],xyz[c[1]][1],xyz[c[1]][2]);
            var v3 = new THREE.Vector3(xyz[c[2]][0],xyz[c[2]][1],xyz[c[2]][2]);

            geometry.vertices.push(v1);
            geometry.vertices.push(v2);
            geometry.vertices.push(v3);

            var i3 = 3*i;
            var face = new THREE.Face3( 0+i3, 1+i3, 2+i3);

            //face.color.setHex( Math.random() * 0xffffff );
            //face.color = new THREE.Color(127/255, 0/255, 0/255);

            try {

                var rgb = cmap[clr[i]];
                face.color = new THREE.Color(rgb[0], rgb[1], rgb[2]);
                //face.color.setHSL( 1,1,1);
            } catch(err) {
                face.color.setHSL(1,1,1);
            }
            geometry.faces.push(face);
        }
    );

    geometry.computeFaceNormals();

    var material = new THREE.MeshBasicMaterial( {vertexColors: THREE.FaceColors, side: THREE.DoubleSide });

    var mesh = new THREE.Mesh( geometry, material );

    mesh.updateMatrix();
    mesh.matrixAutoUpdate = false;
    scene.add( mesh );
    axes = new THREE.AxisHelper( 10 );
//    scene.add( axes );

    var helper = new THREE.GridHelper( 10, 5 );
    //helper.setColors( 0x0000ff, 0x808080 ); // blue central line, gray grid
    helper.material.linewidth = 3;
    helper.geometry.rotateX( Math.PI / 2 );
    helper.position.z = - 0;
//    scene.add( helper );


    var vector = new THREE.Vector3( lookat[0], lookat[1], lookat[2] );
    scene.lookAt( vector );

    fitCameraToSelection( camera, controls, [mesh], fitOffset = 1.2 )




    /*
    var linematerial = new THREE.LineBasicMaterial({color: 0x000, linewidth: 1});
    //var linematerial = new THREE.LineBasicMaterial({color: 0xA9A9A9, linewidth: 1});

    var linegeometry = new THREE.Geometry();


    outer.forEach(function (c, i) {
            //console.log(c, i);
            var v1 = new THREE.Vector3(xyz[c[0]][0], xyz[c[0]][1], xyz[c[0]][2]);
            var v2 = new THREE.Vector3(xyz[c[1]][0], xyz[c[1]][1], xyz[c[1]][2]);

            linegeometry.vertices.push(v1);
            linegeometry.vertices.push(v2);

            var line = new THREE.Line(linegeometry, linematerial, THREE.LinePieces);
            scene.add(line);
        }
    );
    */



    var positions = [];
    var colors = [];
    var indices_array = [];

    outer.forEach(function (c, i) {
            //console.log(c, i);
            //var v1 = new THREE.Vector3(xyz[c[0]][0], xyz[c[0]][1], xyz[c[0]][2]);
            //var v2 = new THREE.Vector3(xyz[c[1]][0], xyz[c[1]][1], xyz[c[1]][2]);

            positions.push(XYZ[c[0]][0], XYZ[c[0]][1], XYZ[c[0]][2]);
            positions.push(XYZ[c[1]][0], XYZ[c[1]][1], XYZ[c[1]][2]);
            indices_array.push(2*i, 2*i+1);

            //var line = new THREE.Line(linegeometry, linematerial, THREE.LinePieces);
            //scene.add(line);
        }
    );

    var geometry = new THREE.BufferGeometry();

    geometry.setIndex( new THREE.BufferAttribute( new Uint16Array( indices_array ), 1 ) );
    geometry.addAttribute( 'position', new THREE.BufferAttribute( new Float32Array( positions ), 3 ) );
    //geometry.addAttribute( 'color', new THREE.BufferAttribute( new Float32Array( colors ), 3 ) );
    geometry.computeBoundingSphere();

    var material = new THREE.LineBasicMaterial({color: 0x808080, linewidth: 1});
    //var material = new THREE.LineBasicMaterial({color: 0x393939, linewidth: 1});
    //var material = new THREE.LineBasicMaterial({color: 0x000, linewidth: 1});
    mesh = new THREE.LineSegments( geometry, material );
    scene.add( mesh );


    window.addEventListener( 'resize', onWindowResize, false );

}


function loadModel ( colorMap, numberOfColors, legendLayout ) {

    var lutColors = [];

    lut = new THREE.Lut( colorMap, numberOfColors );

    lut.setMax( -function_max );
    lut.setMin( function_max );

    if ( legendLayout ) {

            var legend;

            if ( legendLayout == 'horizontal' ) {

                legend = lut.setLegendOn( { 'layout':'horizontal',
                                            'position': { 'x': 0, 'y': 0, 'z': 0 },
                                            'dimensions': {'width':0.5, 'height':3} } );

            } else {

                legend = lut.setLegendOn({'position': { 'x': 0, 'y': 0, 'z': 0 }});

            }

            scene2.add ( legend );

//          var labels = lut.setLegendLabels( { 'title': function_name, 'um': function_unit, 'ticks': 7 } );
            var labels = lut.setLegendLabels( { 'title': function_name, 'ticks': 7 } );

            scene2.add ( labels['title'] );

            for ( var i = 0; i < Object.keys( labels[ 'ticks' ] ).length; i++ ) {

                scene2.add ( labels[ 'ticks' ][ i ] );
                scene2.add ( labels[ 'lines' ][ i ] );

            }

        }

}



function onWindowResize() {

    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();

    camera2.aspect = window.innerWidth / window.innerHeight;
    camera2.updateProjectionMatrix();

    renderer.setSize( window.innerWidth, window.innerHeight );

    render();

}

function animate() {

    requestAnimationFrame( animate );

    controls.update(); // required if controls.enableDamping = true, or if controls.autoRotate = true

    stats.update();

    render();

}

function render() {


    renderer.clear();
    renderer.render( scene, camera );
    renderer.clearDepth();
    renderer.render( scene2, camera2 );



}