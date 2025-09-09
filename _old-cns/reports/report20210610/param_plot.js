function test(y,z){
    var xs = [];
    var x_min = -5;
    var x_max = 5;
    var n = 21;

    var ret = [];
    for(var i=0; i<n; i++){
        xs.push((x_max-x_min)/n * i)
        ret.push(  y*xs[i]*xs[i]+z );
    }
    return {x: xs, y: ret}
};

var data = [test(2,-5)];
Plotly.newPlot('slider-plot', data, 
    sliders: [{
        steps:
    ]}
);

data = [test(1,0)];

document.getElementById('slider-plot').on('plotly_click', function adjustValue1(value)
{
    Plotly.restyle('slider-plot', data);
    alert('test');
});

/*
Plotly.newPlot('slider-plot', [{'x':xs, 'y':ret}]);
slider_plot = document.getElementById('slider-plot');
var config = {responsive: true}
Plotly.newPlot('slider-plot', [{'x':[0], 'y':[0]}], {
  sliders: [{
    pad: {t: 30},
    currentvalue: {
      xanchor: 'left',
      prefix: ': ',
      font: {
        color: '#888',
        size: 20
      }
    },
    steps: [{
      label: 'red',
      method: 'test',
      args: ['line.color', 'red']
    }, {
      label: 'green',
      method: 'restyle',
      args: ['line.color', 'green']
    }, {
      label: 'blue',
      method: 'restyle',
      args: ['line.color', 'blue']
    }]
  },

{
    pad: {t: 100},
    currentvalue: {
      xanchor: 'right',
      prefix: 'color: ',
      font: {
        color: '#888',
        size: 20
      }
    },
    steps: [
        {
          label: 'red',
          method: 'restyle',
          args: ['line.color', 'red']
        }, {
          label: 'green',
          method: 'restyle',
          args: ['line.color', 'green']
        }, {
          label: 'blue',
          method: 'restyle',
          args: ['line.color', 'blue']
        }
    ]
  }
]});
*/
