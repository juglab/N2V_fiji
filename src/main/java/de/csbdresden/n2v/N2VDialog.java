package de.csbdresden.n2v;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.lang.reflect.InvocationTargetException;
import java.util.Collections;
import java.util.List;

import javax.swing.BorderFactory;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JProgressBar;
import javax.swing.LookAndFeel;
import javax.swing.SwingConstants;
import javax.swing.UIManager;
import javax.swing.WindowConstants;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.axis.NumberTickUnit;
import org.jfree.chart.axis.ValueAxis;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYItemRenderer;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.chart.ui.RectangleInsets;
import org.jfree.data.xy.VectorDataItem;
import org.jfree.data.xy.VectorSeries;
import org.jfree.data.xy.VectorSeriesCollection;

public class N2VDialog {

	private final static int DEFAULT_WIDTH = 500;
	private final static int DEFAULT_MIN_HEIGHT = 150;
	private final static int DEFAULT_MAX_HEIGHT = 650;
	private final static int DEFAULT_CHART_HEIGHT = 300;
	private final static int DEFAULT_BAR_WIDTH = 400;
	private final static int DEFAULT_BAR_HEIGHT = 40;
	private final static String FRAME_TITLE = "N2V for Fiji";
	private final static String CHART_TITLE = "Epochal Losses";
	private final static String XAXIS_LABEL = "Epoch";
	private final static String YAXIS_LABEL = "Loss";

	private ChartPanel chartPanel = null;
	private int nEpochSteps;
	private VectorSeries averageLossData = null;
	private VectorSeries validationLossData = null;
	private JProgressBar progressBar;
	private int nEpochs;
	private JFrame frame;
	private JPanel topPanel;
	private N2VTraining n2v;
	private VectorSeriesCollection data;
	private XYPlot plot;
	private JLabel progressSpinner;
	private Color currentColor = Color.BLUE;

	public N2VDialog(N2VTraining n2v) {
		this.n2v = n2v;
		createProgressPanel();
		createChartPanel();

		frame = new JFrame( FRAME_TITLE );
		try {
			javax.swing.SwingUtilities.invokeAndWait(
					new Runnable() {

						@Override
						public void run() {
							frame.setSize( DEFAULT_WIDTH+20, DEFAULT_MIN_HEIGHT );
							frame.setDefaultCloseOperation( WindowConstants.DISPOSE_ON_CLOSE );
							frame.setLayout( new BorderLayout() );

							frame.add( topPanel, BorderLayout.NORTH );
							frame.add( chartPanel, BorderLayout.CENTER );
							chartPanel.setVisible(false);
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

	private void createProgressPanel() {
		// Progress panel
		topPanel = new JPanel();
		topPanel.setPreferredSize( new Dimension( DEFAULT_WIDTH, DEFAULT_BAR_HEIGHT*3 ) );
		topPanel.setBorder( BorderFactory.createEmptyBorder( 5, 5, 5, 5 ) );
		topPanel.setLayout( new GridBagLayout() );

		JButton cancelBtn = new JButton( "Cancel Training" );
		cancelBtn.addActionListener( new ActionListener() {

			@Override
			public void actionPerformed( ActionEvent e ) {
				n2v.cancelTraining();
			}

		} );

		ImageIcon animatedIcon = new ImageIcon( N2VDialog.class.getClassLoader().getResource( "hard-workout.gif" ) );
		progressSpinner = new JLabel( "", animatedIcon, JLabel.CENTER );
		LookAndFeel lf = UIManager.getLookAndFeel();
		UIManager.put( "ProgressBarUI", "javax.swing.plaf.metal.MetalProgressBarUI" );
		progressBar = new JProgressBar( SwingConstants.HORIZONTAL );
		progressBar.setPreferredSize( new Dimension( DEFAULT_BAR_WIDTH, DEFAULT_BAR_HEIGHT ) );
		progressBar.setStringPainted( true );

		// Place components
		GridBagConstraints gbc = new GridBagConstraints();
		gbc.anchor = GridBagConstraints.CENTER;
		gbc.fill = GridBagConstraints.NONE;
		gbc.insets = new Insets( 5, 5, 5, 5 );
		gbc.gridy = 0;
		topPanel.add( cancelBtn, gbc );

		gbc.gridy = 1;
		topPanel.add( progressSpinner, gbc );

		gbc.gridy = 2;
		gbc.fill = GridBagConstraints.BOTH;
		topPanel.add( progressBar, gbc );
		progressBar.setVisible( false );

	}

	private void createChartPanel() {

		// Basic chart layout
		data = new VectorSeriesCollection();
		JFreeChart chart = ChartFactory.createTimeSeriesChart( CHART_TITLE, XAXIS_LABEL, YAXIS_LABEL, data );
		chart.setBackgroundPaint( Color.WHITE );
		chartPanel = new ChartPanel( chart, false );
		chartPanel.setBorder( BorderFactory.createEmptyBorder( 5, 5, 5, 5 ) );
		chartPanel.setPreferredSize( new Dimension( DEFAULT_WIDTH, DEFAULT_CHART_HEIGHT ) );
		chartPanel.setFillZoomRectangle( true );
		chartPanel.setMouseWheelEnabled( true );

		plot = ( XYPlot ) chart.getPlot();
		plot.setBackgroundPaint( Color.LIGHT_GRAY );
		plot.setDomainGridlinePaint( Color.WHITE );
		plot.setRangeGridlinePaint( Color.WHITE );
		plot.setAxisOffset( new RectangleInsets( 5.0, 5.0, 5.0, 5.0 ) );
		plot.setDomainCrosshairVisible( true );
		plot.setRangeCrosshairVisible( true );
		plot.setDomainAxis( new NumberAxis() );
		plot.setRangeAxis( new NumberAxis() );
		plot.getRangeAxis().setAutoRange(true);

		XYItemRenderer r = plot.getRenderer();
		if ( r instanceof XYLineAndShapeRenderer ) {
			XYLineAndShapeRenderer renderer = ( XYLineAndShapeRenderer ) r;
			renderer.setDefaultShapesVisible( true );
			renderer.setDefaultShapesFilled( true );
			renderer.setDrawSeriesLineAsPath( true );
		}

	}

	public void initChart( int nEpochs, int nEpochSteps ) {
		this.nEpochSteps = nEpochSteps;
		this.nEpochs = nEpochs;
		averageLossData = new VectorSeries( "Average Loss" );
		validationLossData = new VectorSeries( "Validation Loss" );
		data.addSeries( averageLossData );
		data.addSeries( validationLossData );
		progressSpinner.setVisible( false );
		progressBar.setVisible( true );
		chartPanel.setVisible( true );
		frame.revalidate();
		frame.pack();
	}

	public void updateChart( int nEpoch, List< Double > losses, double validationLoss ) {

		double averageLoss = 0.0;
		for ( int i = 0; i < losses.size(); i++ ) {
			averageLoss += losses.get( i );
		}
		averageLoss = averageLoss / nEpochSteps;

		if ( nEpoch == 1 ) {
			// Size axis to zoom onto first epoch data
			NumberAxis xAxis = ( NumberAxis ) plot.getDomainAxis();
			xAxis.setStandardTickUnits( NumberAxis.createIntegerTickUnits() );
			xAxis.setRange( 1.0, nEpochs );
			xAxis.setTickUnit( new NumberTickUnit( 10 ) );
		}

		averageLossData.add( new VectorDataItem( ( double ) nEpoch, averageLoss, 0.0, 0.0 ), true );
		validationLossData.add( new VectorDataItem( ( double ) nEpoch, validationLoss, 0.0, 0.0 ), true );
		chartPanel.revalidate();
		chartPanel.repaint();

	}

	public void updateProgressText( String text ) {

		try {
			javax.swing.SwingUtilities.invokeAndWait(
					new Runnable() {

						@Override
						public void run() {
							progressSpinner.setText( text );
						}
					} );
		} catch ( InterruptedException e ) {
			e.printStackTrace();
		} catch ( InvocationTargetException e ) {
			e.printStackTrace();
		}
	}

	public void updateProgress( int epoch, int step ) {
		int maxBareSize = 1; // 10unit for 100%
		int remainPercent = ( ( 100 * step ) / nEpochSteps ) / maxBareSize;
		progressBar.setValue( remainPercent );
		progressBar.setString( "Epoch " + epoch + "/" + nEpochs + ", step " + step + "/" + nEpochSteps );
	}
}
