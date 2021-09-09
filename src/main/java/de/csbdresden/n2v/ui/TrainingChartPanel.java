/*-
 * #%L
 * N2V plugin
 * %%
 * Copyright (C) 2019 - 2020 Center for Systems Biology Dresden
 * %%
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * #L%
 */
package de.csbdresden.n2v.ui;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.annotations.XYTitleAnnotation;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.axis.ValueAxis;
import org.jfree.chart.block.BlockBorder;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYItemRenderer;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.chart.title.LegendTitle;
import org.jfree.chart.ui.RectangleAnchor;
import org.jfree.chart.ui.RectangleEdge;
import org.jfree.chart.ui.RectangleInsets;
import org.jfree.data.xy.VectorDataItem;
import org.jfree.data.xy.VectorSeries;
import org.jfree.data.xy.VectorSeriesCollection;

import javax.swing.*;
import javax.swing.border.EmptyBorder;
import java.awt.*;
import java.net.URL;
import java.util.Collections;
import java.util.List;

public class TrainingChartPanel extends JPanel {

	private final static int DEFAULT_CHART_HEIGHT = 400;
	private final static int DEFAULT_CHART_WIDTH = 600;

	private final static String CHART_TITLE = "";
	private final static String XAXIS_LABEL = "Epoch";
	private final static String YAXIS_LABEL = "Loss";

	private ChartPanel chartPanel;
	private XYPlot plot;
	private WaitLayerUI waitLayer;

	private int nEpochSteps;
	private VectorSeriesCollection data;
	private VectorSeries averageLossData;
	private VectorSeries validationLossData;
	private TrainingProgressPanel progressPanel;

	private final static ImageIcon waitingIcon = new ImageIcon( TrainingProgress.class.getClassLoader().getResource( "hard-workout.gif" ) );

	public TrainingChartPanel(int nEpochs, int nEpochSteps ) {

		setLayout( new BorderLayout() );
		setBackground( Color.WHITE );
		setBorder(new EmptyBorder(10,10,10,10));
		

		
		this.nEpochSteps = nEpochSteps;

		waitLayer = new WaitLayerUI(this);

		// Progress bars
		progressPanel = new TrainingProgressPanel( nEpochs, nEpochSteps );
		progressPanel.setBorder(BorderFactory.createEmptyBorder(0, 0, 0, 4));
		add( progressPanel, BorderLayout.NORTH );

		//Chart
		data = new VectorSeriesCollection();

		JFreeChart chart = ChartFactory.createTimeSeriesChart( CHART_TITLE, XAXIS_LABEL, YAXIS_LABEL, data );
		chart.setBackgroundPaint( Color.WHITE );
		chart.removeLegend();

		chartPanel = new ChartPanel( chart, false );
		chartPanel.setBorder( BorderFactory.createEmptyBorder( 2, 2, 2, 2 ) );
		chartPanel.setPreferredSize( new Dimension( DEFAULT_CHART_WIDTH, DEFAULT_CHART_HEIGHT ) );
		chartPanel.setFillZoomRectangle( true );
		chartPanel.setMouseWheelEnabled( true );
		chartPanel.setBackground( Color.WHITE );
		add( chartPanel, BorderLayout.CENTER );

		plot = ( XYPlot ) chart.getPlot();
		plot.setBackgroundPaint( Color.LIGHT_GRAY );
		plot.setDomainGridlinePaint( Color.WHITE );
		plot.setRangeGridlinePaint( Color.WHITE );
		plot.setAxisOffset( new RectangleInsets( 5.0, 5.0, 5.0, 5.0 ) );
		plot.setDomainCrosshairVisible( true );
		plot.setRangeCrosshairVisible( true );
		plot.setDomainAxis( new NumberAxis() );
		plot.setRangeAxis( new NumberAxis() );

		NumberAxis xAxis = ( NumberAxis ) plot.getDomainAxis();
		xAxis.setStandardTickUnits( NumberAxis.createIntegerTickUnits() );
		xAxis.setRange( Math.min(nEpochs-1, 1.0), nEpochs);

		XYItemRenderer r = plot.getRenderer();
		if ( r instanceof XYLineAndShapeRenderer ) {
			XYLineAndShapeRenderer renderer = ( XYLineAndShapeRenderer ) r;
			renderer.setDefaultShapesVisible( true );
			renderer.setDefaultShapesFilled( true );
			renderer.setDrawSeriesLineAsPath( true );
		}
		
		XYPlot plot = (XYPlot) chart.getPlot();
		LegendTitle lt = new LegendTitle(plot);
		lt.setItemFont(new Font("Dialog", Font.PLAIN, 12));
		lt.setBackgroundPaint(new Color(200, 200, 255, 100));
		lt.setFrame(new BlockBorder(Color.white));
		lt.setPosition(RectangleEdge.BOTTOM);
		XYTitleAnnotation ta = new XYTitleAnnotation(0.98, 0.98, lt,RectangleAnchor.TOP_RIGHT);

		ta.setMaxWidth(0.48);
		plot.addAnnotation(ta);
		

		// Data 
		averageLossData = new VectorSeries( "Training Loss" );
		validationLossData = new VectorSeries( "Validation Loss" );
		data.addSeries( averageLossData );
		data.addSeries( validationLossData );

		waitLayer.start();
	}

	public JComponent getPanel() {
		return new JLayer< Container >( this, waitLayer );
	}

	public void updateProgress( int epoch, int step ) {
		if ( epoch == 1 ) {
			waitLayer.stop();
			repaint();
		}
		progressPanel.updateProgress( epoch, step );
	}

	public void updateChart( int nEpoch, List< Double > losses, double validationLoss ) {

		double averageLoss = 0.0;
		for ( int i = 0; i < losses.size(); i++ ) {
			averageLoss += losses.get( i );
		}
		averageLoss = averageLoss / nEpochSteps;

		if ( nEpoch == 1 ) {
			waitLayer.stop();
			// Size axis to zoom onto first epoch data
			double ymax = Collections.max( losses );
			double ymin = Collections.min( losses );
			ValueAxis yAxis = plot.getRangeAxis();
			yAxis.setRange( Math.floor( ymin ) - 0.1, Math.ceil( ymax ) + 0.1 );
		}

		averageLossData.add( new VectorDataItem( ( double ) nEpoch, averageLoss, 0.0, 0.0 ), true );
		validationLossData.add( new VectorDataItem( ( double ) nEpoch, validationLoss, 0.0, 0.0 ), true );
		chartPanel.repaint();

	}

	public void setWaitingIcon(URL iconUrl, float iconScale, int iconAlignment, int iconOffsetX, int iconOffsetY) {
		waitLayer.setWaitingIcon(iconUrl, iconScale, iconAlignment, iconOffsetX, iconOffsetY);
	}
}
