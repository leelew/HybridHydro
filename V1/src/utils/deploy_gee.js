/**Map layer configuration
 */
 
//-----------------------------
//CONSTANTS (NOTES: changeable)
//-----------------------------
var begin_date = ee.Date.fromYMD(2017, 4, 1);
var end_date = ee.Date.fromYMD(2018, 4, 1);
var geometry = ee.Geometry.Polygon([[[3, 73],
                                     [3, 135],
                                     [53, 73],
                                     [53, 135]]], null, false);
var palettes = require('users/gena/packages:palettes'); 
var palette = palettes.misc.jet[7];//misc.jet[7];
var vis = {min: 0, max: 30, palette: palette};

//---------------------------------------------
//predicted surface soil moisture (TODO:(luli))
//---------------------------------------------
var pred = ee.Image("users/SysuLewLee1/testImage").visualize(vis);
print(pred)
//----------------------------------------------------
//observation (TODO:add different sources observation)
//----------------------------------------------------
// gldas
var gldas = ee.ImageCollection('NASA/GLDAS/V021/NOAH/G025/T3H').filterDate(begin_date, end_date);
var gldas_ssm = gldas.select('SoilMoi0_10cm_inst');
gldas_ssm = gldas_ssm.map(function(image){
  return image.rename("GLDAS_0-10cm")});

// smap enhanced l3
var smap = ee.ImageCollection("NASA_USDA/HSL/SMAP10KM_soil_moisture").filterDate(begin_date,end_date);
var smap_ssm = smap.select(['ssm']);
smap_ssm = smap_ssm.map(function(image){
  return image.rename("SMAP_L3_0-2cm")});

// mean smap enhanced l3 (TODO: removed after trial version)
var mean_smap_ssm = smap_ssm.mean().visualize(vis);

// Near-real-time smap l3
//list = smap_ssm.toList()
//var smap_nrt = ee.Image(list.get(0))


//-----------
//merge (TODO)
//-----------
var sm_all = smap_ssm.merge(gldas_ssm);
var sm_all_pred = sm_all.merge(pred);
print(sm_all_pred)


/** Map */
 
 
var mapPanel = ui.Map();
var layers = mapPanel.layers();

// create layers
var mean_smap_ssm_layer = ui.Map.Layer(mean_smap_ssm).setName('mean smap enhanced l3');
var predict_layer = ui.Map.Layer(pred).setName('predict sm');
var clip_geometry_layer = ui.Map.Layer(geometry).setName('click geometry');

// palettes setting from \ee-palettes\
// add layers
//mapPanel.layers().set(2, mean_smap_ssm_layer);
mapPanel.layers().set(2, predict_layer);
mapPanel.layers().set(3, mean_smap_ssm_layer);


//layers.add(mean_smap_ssm_layer, 'mean smap enhanced l3'); //(FIXME: colormap bug)
//layers.add(clip_geometry_layer); //(FIXME: clip layer bug)
//layers.add(predict_layer,  'predict')
print(mean_smap_ssm)






/*
 * Panel setup
 */

// Create a panel to hold title, intro text, chart and legend components.
var inspectorPanel = ui.Panel({style: {width: '37%'}});

// Create an intro panel with labels.
var intro = ui.Panel([
  ui.Label({
    value: 'High Resolution Soil Environment Prediction Platform',
    style: {fontSize: '20px', fontWeight: 'bold', fontFamily: 'serif', position: 'top-center'}
  }),
  ui.Label({
    value: 'I. Random Visualization',
    style: {fontSize: '18px', fontWeight: 'bold', fontFamily: 'serif', position: 'top-center'}
  }),
  ui.Label({
    value: 'Click a location to see its time series of soil moisture.',
    style: {fontFamily: 'serif', position: 'top-center'}
  })
]);

// Create panels to hold lon/lat values.
var lon = ui.Label(); 
var lat = ui.Label();



/*
 * Panel 2
 */

var intro_download = ui.Panel([
  ui.Label({
    value: 'II. Download HRSEPP data',
    style: {fontSize: '18px', fontWeight: 'bold', fontFamily: 'serif', position: 'top-center'}
  }),
  ui.Label({
    value: 'Choose and download HRSEPP soil moisture.',
    style: {fontFamily: 'serif', position: 'top-center'}
  })
])

var filters = {
  startDate: ui.Textbox('YYYY-MM-DD', '2018-03-01'),
  endDate: ui.Textbox('YYYY-MM-DD', '2018-05-11'),
  applyButton: ui.Button({
    label: 'download', 
    onClick: function(){
      var Images = ee.ImageCollection("NASA_USDA/HSL/SMAP10KM_soil_moisture").select(['ssm']);
      
      // Set filter variables.
      var start = filters.startDate.getValue();
      if (start) start = ee.Date(start);
      var end = filters.endDate.getValue();
      if (end) end = ee.Date(end);
      if (start) Images = Images.filterDate(start, end);

      //(FIXME: export to google drive)
      Export.image.toDrive({
          image: Images,
    })
  }
}),
  loadingLabel: ui.Label({
    value: 'downloading...',
    style: {stretch: 'vertical', color: 'gray', shown: false}
  })
}


// filter control widgets
filters.panel = ui.Panel({
  widgets: [
    ui.Label('Start Date'), 
    filters.startDate,
    ui.Label('End Date'),
    filters.endDate,
    ui.Panel([
      filters.applyButton,
      filters.loadingLabel])
  ]})


// Add module for panel
inspectorPanel.add(intro);
inspectorPanel.add(ui.Panel([lon, lat], ui.Panel.Layout.flow('horizontal')));
inspectorPanel.add(ui.Label('[Chart]')); // Add placeholders for the chart and legend.
//inspectorPanel.add(ui.Label('[Legend]'));
inspectorPanel.add(intro_download);
inspectorPanel.add(filters.panel)



/*
 * Chart setup
 */

// Generates a new time series chart of SM for the given coordinates.
var generateChart = function (coords) {
  // Update the lon/lat panel with values from the click event.
  lon.setValue('lon: ' + coords.lon.toFixed(2)).style().set({fontFamily: 'serif'});
  lat.setValue('lat: ' + coords.lat.toFixed(2)).style().set({fontFamily: 'serif'});

  // Add a dot for the point clicked on.
  var point = ee.Geometry.Point(coords.lon, coords.lat);
  var dot = ui.Map.Layer(point, {color: '000000'}, 'clicked location');
  // Add the dot as the second layer, so it shows up on top of the composite.
  mapPanel.layers().set(1, dot); //TODO: API of .set()

  // Make a chart from the time series.
  var smChart = ui.Chart.image.series(sm_all, point, ee.Reducer.mean(), 500); // reducer.mean() is default setting

  // Customize the chart.
  smChart.setOptions({
    title: 'surface soil moisture time series',
    vAxis: {title: '(%)'},
    hAxis: {title: 'Date', format: 'YY-MM-yy', gridlines: {count: 7}},
    //interpolateNulls: true,
    series: {
      0: {
        color: 'red',
        lineWidth: 0,
        pointsVisible: true,
        pointSize: -1,
      },
      1: {
        color: 'blue',
        lineWidth: 0,
        pointsVisible: true,
        pointSize: 2
      },
      2: {
        color: 'green',
        lineWidth: 0,
        pointsVisible: true,
        pointSize: 2
      }
    },
    legend: {position: 'right'},
  });
  
  smChart.setDownloadable('png')
  // Add the chart at a fixed position, so that new charts overwrite older ones.
  inspectorPanel.widgets().set(2, smChart);
};




/*
 * Legend setup
 */

// Creates a color bar thumbnail image for use in legend from the given color
// palette.
function makeColorBarParams(palette) {
  return {
    bbox: [0, 0, 1, 0.1],
    dimensions: '100x10',
    format: 'png',
    min: 0,
    max: 1,
    palette: palette,
  };
}

// Create the color bar for the legend.
var colorBar = ui.Thumbnail({
  image: ee.Image.pixelLonLat().select(0),
  params: makeColorBarParams(vis.palette),
  style: {stretch: 'horizontal', margin: '0px 8px', maxHeight: '16px'},
});

// Create a panel with three numbers for the legend.
var legendLabels = ui.Panel({
  widgets: [
    ui.Label(vis.min, {margin: '4px 8px'}),
    ui.Label(
        (vis.max / 2),
        {margin: '4px 8px', textAlign: 'center', stretch: 'horizontal'}),
    ui.Label(vis.max, {margin: '4px 8px'})
  ],
  layout: ui.Panel.Layout.flow('horizontal')
});

var legendTitle = ui.Label({
  value: 'Tomorrow surface soil moisture (%)',
  style: {fontWeight: 'bold', textAlign: 'center'}
});

var legendPanel = ui.Panel([legendTitle, colorBar, legendLabels]);
mapPanel.add(legendPanel)




// Register a callback on the default map to be invoked when the map is clicked.
mapPanel.onClick(generateChart);

// Configure the map.
mapPanel.style().set('cursor', 'crosshair');

// Initialize with a test point at SYSU.
var initialPoint = ee.Geometry.Point(113, 23);
mapPanel.centerObject(initialPoint, 4);



/*
 * Initialize the app
 */

// Replace the root with a SplitPanel that contains the inspector and map.
ui.root.clear();
ui.root.add(ui.SplitPanel(inspectorPanel, mapPanel));

generateChart({
  lon: initialPoint.coordinates().get(0).getInfo(),
  lat: initialPoint.coordinates().get(1).getInfo()
});


