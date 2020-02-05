package de.csbdresden.n2v;

import java.awt.BorderLayout;
import java.awt.Color;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JProgressBar;
import javax.swing.JTextField;
import javax.swing.SwingConstants;
import javax.swing.WindowConstants;

import org.knowm.xchart.XChartPanel;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYSeries;
import org.knowm.xchart.style.colors.XChartSeriesColors;
import org.knowm.xchart.style.lines.SeriesLines;
import org.knowm.xchart.style.markers.SeriesMarkers;

public class N2VDialog {

	private final static int DEFAULT_WIDTH = 600;
	private final static int DEFAULT_HEIGHT = 400;
	private static String title = "N2V for Fiji";
	private static String chartTitle = "Epochal Losses";
	private static String xAxisTitle = "Epoch";
	private static String yAxisTitle = "Loss";

	private XYChart chart = null;
	private XChartPanel< XYChart > chartPanel = null;
	private int nEpochSteps;
	List< Double > epochData = null;
	List< Double > averageLossData = null;
	List< Double > validationLossData = null;
	protected JProgressBar progressBar;

	/**
	 * Default constructor for a 600x400 dialog
	 */
	public N2VDialog( int nEpochs, int nEpochSteps ) {
		this( DEFAULT_WIDTH, DEFAULT_HEIGHT, nEpochs, nEpochSteps );
	}

	public N2VDialog( int dialogWidth, int dialogHeight, int nEpochs, int nEpochSteps ) {
		final JFrame frame = new JFrame( title );
		try {
			javax.swing.SwingUtilities.invokeAndWait(
					new Runnable() {

						@Override
						public void run() {
							frame.setDefaultCloseOperation( WindowConstants.EXIT_ON_CLOSE );
							frame.setSize( dialogWidth, dialogHeight );
							frame.setLayout( new BorderLayout() );

							epochData = new ArrayList< Double >( nEpochs );
							for ( int i = 0; i < nEpochs; i++ ) {
								epochData.add( i + 1.0 );
							}
							averageLossData = new ArrayList<>();
							validationLossData = new ArrayList<>();

							// Create chart with as much info as possible
							chart = new XYChart( dialogWidth, dialogHeight );
							chart.setTitle( chartTitle );
							chart.setXAxisTitle( xAxisTitle );
							chart.setYAxisTitle( yAxisTitle );
							chart.getStyler().setXAxisMin( 1.0 );
							chart.getStyler().setXAxisMax( ( double ) nEpochs );
							chartPanel = new XChartPanel< XYChart >( chart );
							// Create yaxis scale panel
							JPanel scalePanel = new JPanel();
							scalePanel.add(new JTextField(5));
							scalePanel.add(new JTextField(5));

							progressBar = new JProgressBar( SwingConstants.VERTICAL );
							frame.add( "North", scalePanel );
							frame.add( "Center", chartPanel );
							frame.add( "South", progressBar );
							frame.pack();
							frame.setLocationRelativeTo( null );
							frame.setVisible( true );
						}
					} );
		} catch ( InterruptedException e ) {
			e.printStackTrace();
		} catch ( InvocationTargetException e ) {
			e.printStackTrace();
		}

	}

	public void update( int nEpoch, List< Double > losses, double validationLoss ) {

		double averageLoss = 0.0;
		for ( int i = 0; i < losses.size(); i++ ) {
			averageLoss += losses.get( i );
		}

		averageLossData.add( averageLoss / nEpochSteps );
		validationLossData.add( validationLoss );

		if ( nEpoch == 1 ) {
			// Size axis to zoom onto first epoch data
			double ymax = Collections.max( losses );
			double ymin = Collections.min( losses );
			chart.getStyler().setYAxisMin( Math.floor( ymin ) - 0.5 );
			chart.getStyler().setYAxisMax( Math.ceil( ymax ) + 0.5 );
			XYSeries series1 = chart.addSeries( "Average Loss", epochData, averageLossData );
			series1.setLineColor( XChartSeriesColors.BLUE );
			series1.setMarkerColor( Color.BLUE );
			series1.setMarker( SeriesMarkers.CIRCLE );
			series1.setLineStyle( SeriesLines.SOLID );
			XYSeries series2 = chart.addSeries( "Validation Loss", epochData, validationLossData );
			series2.setLineColor( XChartSeriesColors.GREEN );
			series2.setMarkerColor( Color.GREEN );
			series2.setMarker( SeriesMarkers.DIAMOND );
			series2.setLineStyle( SeriesLines.SOLID );
			chartPanel.revalidate();
			chartPanel.repaint();
		} else {

			chart.updateXYSeries( "Average Loss", epochData, averageLossData, null );
			chart.updateXYSeries( "Validation Loss", epochData, validationLossData, null );
			chartPanel.revalidate();
			chartPanel.repaint();
		}

	}

}
