package de.csbdresden.n2v;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import javax.swing.InputVerifier;
import javax.swing.JButton;
import javax.swing.JComponent;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JProgressBar;
import javax.swing.JSplitPane;
import javax.swing.JTextField;
import javax.swing.SwingConstants;
import javax.swing.WindowConstants;

import org.knowm.xchart.XChartPanel;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYSeries;
import org.knowm.xchart.style.colors.XChartSeriesColors;
import org.knowm.xchart.style.lines.SeriesLines;
import org.knowm.xchart.style.markers.SeriesMarkers;

import net.miginfocom.swing.MigLayout;

public class N2VDialog {

	private final static int DEFAULT_WIDTH = 600;
	private final static int DEFAULT_HEIGHT = 100;
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
	private int nEpochs;
	protected JSplitPane splitPane;
	private JFrame frame;
	protected JLabel message;
	protected JButton rescaleBtn;

	public N2VDialog( final N2V n2v) {
		frame = new JFrame( title );
		try {
			javax.swing.SwingUtilities.invokeAndWait(
					new Runnable() {

						@Override
						public void run() {
							frame.setDefaultCloseOperation( WindowConstants.EXIT_ON_CLOSE );
							splitPane = new JSplitPane( JSplitPane.VERTICAL_SPLIT );
							splitPane.setContinuousLayout( true );

							// Pogress panel
							JPanel topPanel = new JPanel();
							topPanel.setLayout( new MigLayout("","[center]","[center]") );
							message = new JLabel();
							topPanel.add( message, "align center, wrap" );
							progressBar = new JProgressBar( SwingConstants.HORIZONTAL );
							progressBar.setPreferredSize( new Dimension( 400, 20 ) );
							progressBar.setIndeterminate( true );
							progressBar.setVisible( true );
							topPanel.add( progressBar, "align center, wrap" );
							JButton cancelBtn = new JButton( "Cancel Training" );
							cancelBtn.addActionListener( new ActionListener() {

								@Override
								public void actionPerformed( ActionEvent e ) {
									if (n2v.cancelTraining()) {
										//TODO reset UI
									}
								}

							} );
							topPanel.add( cancelBtn, "align center" );
							splitPane.setTopComponent( topPanel );

							// Create chart with as much info as possible
							JPanel bottomPanel = new JPanel();
							bottomPanel.setLayout( new BorderLayout() );
							chart = new XYChart( DEFAULT_WIDTH, DEFAULT_HEIGHT * 4 );
							chart.setTitle( chartTitle );
							chart.setXAxisTitle( xAxisTitle );
							chart.setYAxisTitle( yAxisTitle );
							chartPanel = new XChartPanel< XYChart >( chart );
							bottomPanel.add( chartPanel, BorderLayout.CENTER );

							// YAxis rescaling
							JPanel scalePanel = new JPanel();
							scalePanel.add( new JLabel( "ymin" ) );
							JTextField yminTF = new JTextField( 5 );
							yminTF.setInputVerifier( new TFInputVerifier() );
							scalePanel.add( yminTF );
							scalePanel.add( new JLabel( "ymax" ) );
							JTextField ymaxTF = new JTextField( 5 );
							ymaxTF.setInputVerifier( new TFInputVerifier() );
							scalePanel.add( ymaxTF );
							rescaleBtn = new JButton( "Rescale" );
							rescaleBtn.addActionListener( new ActionListener() {

								@Override
								public void actionPerformed( ActionEvent e ) {
									//TODO validate input is not garbage.
									chart.getStyler().setYAxisMin( Double.valueOf( yminTF.getText() ) );
									chart.getStyler().setYAxisMax( Double.valueOf( ymaxTF.getText() ) );
									chartPanel.revalidate();
									chartPanel.repaint();
								}

							} );
							scalePanel.add( rescaleBtn );
							bottomPanel.add( scalePanel, BorderLayout.SOUTH );

							splitPane.setBottomComponent( bottomPanel );
							splitPane.getBottomComponent().setVisible( false );

							frame.add( splitPane );
							frame.setSize( DEFAULT_WIDTH, DEFAULT_HEIGHT );
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

	public void initChart( int nEpochs, int nEpochSteps ) {
		this.nEpochSteps = nEpochSteps;
		this.nEpochs = nEpochs;
		epochData = new ArrayList< Double >();
		averageLossData = new ArrayList<>();
		validationLossData = new ArrayList<>();
	}

	public void updateChart( int nEpoch, List< Double > losses, double validationLoss ) {

		double averageLoss = 0.0;
		for ( int i = 0; i < losses.size(); i++ ) {
			averageLoss += losses.get( i );
		}

		epochData.add( ( double ) nEpoch );
		averageLossData.add( averageLoss / nEpochSteps );
		validationLossData.add( validationLoss );

		if ( nEpoch == 1 ) {
			// Size axis to zoom onto first epoch data
			double ymax = Collections.max( losses );
			double ymin = Collections.min( losses );
			chart.getStyler().setXAxisMin( 1.0 );
			chart.getStyler().setXAxisMax( ( double ) nEpochs );
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
			splitPane.getBottomComponent().setVisible( true );
			frame.setSize( 800, 600 );
		} else {

			chart.updateXYSeries( "Average Loss", epochData, averageLossData, null );
			chart.updateXYSeries( "Validation Loss", epochData, validationLossData, null );
			chartPanel.revalidate();
			chartPanel.repaint();
		}

	}

	public void updateProgress(String text) {
		message.setText(text);
		message.repaint();
	}
	
	public void updateProgress(int epoch, int step ) {
		int maxBareSize = 10; // 10unit for 100%
		int remainProcent = ( ( 100 * step ) / nEpochSteps ) / maxBareSize;
		char defaultChar = '-';
		String icon = "*";
		String bare = new String( new char[ maxBareSize ] ).replace( '\0', defaultChar ) + "]";
		StringBuilder bareDone = new StringBuilder();
		bareDone.append( "[" );
		for ( int i = 0; i < remainProcent; i++ ) {
			bareDone.append( icon );
		}
		String bareRemain = bare.substring( remainProcent );
		message.setText( "Epoch " + epoch + "/" + nEpochs + ", step " + step + "/" + nEpochSteps + " \t" + bareDone + bareRemain);
		message.repaint();
	}

	private class TFInputVerifier extends InputVerifier {

		@Override
		public boolean verify( JComponent input ) {
			String text = ( ( JTextField ) input ).getText();
			try {
				new Double( text );
				( ( JTextField ) input ).setForeground( Color.BLACK );
				rescaleBtn.setEnabled( true );
				return true;
			} catch ( NumberFormatException e ) {
				( ( JTextField ) input ).setForeground( Color.RED );
				rescaleBtn.setEnabled( false );
				return false;
			}
		}
	}
}
