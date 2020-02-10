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
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import javax.swing.BorderFactory;
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
	private final static int DEFAULT_MIN_HEIGHT = 150;
	private final static int DEFAULT_MAX_HEIGHT = 650;
	private final static int DEFAULT_CHART_HEIGHT = 400;
	private final static int DEFAULT_BAR_WIDTH = 400;
	private final static int DEFAULT_BAR_HEIGHT = 20;
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
	protected JPanel bottomPanel;
	protected JPanel topPanel;
	private N2V n2v;

	public N2VDialog( final N2V n2v) {
		this.n2v = n2v;
		frame = new JFrame( title );
		try {
			javax.swing.SwingUtilities.invokeAndWait(
					new Runnable() {

						@Override
						public void run() {
							frame.setSize( DEFAULT_WIDTH, DEFAULT_MIN_HEIGHT );
							frame.setDefaultCloseOperation( WindowConstants.EXIT_ON_CLOSE );
							frame.setLayout( new BorderLayout() );
								
							createProgressPanel();
							createChartPanel();
							
							frame.add( topPanel, BorderLayout.NORTH );
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
	
	private void createProgressPanel()
	{
		// Progress panel
		topPanel = new JPanel();
		topPanel.setBorder( BorderFactory.createEmptyBorder(5, 5, 10, 5));
		topPanel.setLayout( new GridBagLayout());
	    GridBagConstraints gbc = new GridBagConstraints();
	    gbc.anchor = GridBagConstraints.CENTER;
	    gbc.fill = GridBagConstraints.NONE;
	    gbc.insets = new Insets(5,5,5,5);
	    gbc.gridy = 0;
	    
		message = new JLabel();
		topPanel.add( message, gbc );
		
	    gbc.gridy = 1;
	    gbc.insets = new Insets(0,10,0,10);
	    gbc.fill = GridBagConstraints.HORIZONTAL;
		progressBar = new JProgressBar( SwingConstants.HORIZONTAL );
		progressBar.setPreferredSize( new Dimension(DEFAULT_BAR_WIDTH, DEFAULT_BAR_HEIGHT));
		progressBar.setIndeterminate( true );
		progressBar.setVisible( true );
		topPanel.add( progressBar, gbc );
		
	    gbc.gridy = 2;
	    gbc.insets = new Insets(5,5,5,5);
	    gbc.fill = GridBagConstraints.NONE;
		JButton cancelBtn = new JButton( "Cancel Training" );
		cancelBtn.addActionListener( new ActionListener() {

			@Override
			public void actionPerformed( ActionEvent e ) {
				if (n2v.cancelTraining()) {
					//TODO reset UI
				}
			}

		} );
		topPanel.add( cancelBtn, gbc );
	}
	
	private void createChartPanel()
	{
		bottomPanel = new JPanel();
		bottomPanel.setLayout( new BorderLayout() );
		bottomPanel.setBorder(BorderFactory.createEmptyBorder(5, 5, 10, 5));
		
		// Basic chart layout
		chart = new XYChart( DEFAULT_WIDTH, DEFAULT_CHART_HEIGHT);
		chart.setTitle( chartTitle );
		chart.setXAxisTitle( xAxisTitle );
		chart.setYAxisTitle( yAxisTitle );
		chartPanel = new XChartPanel< XYChart >( chart );
		chartPanel.setPreferredSize( new Dimension(DEFAULT_WIDTH, DEFAULT_CHART_HEIGHT));
		bottomPanel.add( chartPanel, BorderLayout.CENTER );

		// Inputs for re-scaling y axis
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
				chart.getStyler().setYAxisMin( Double.valueOf( yminTF.getText() ) );
				chart.getStyler().setYAxisMax( Double.valueOf( ymaxTF.getText() ) );
				chartPanel.revalidate();
				chartPanel.repaint();
			}

		} );
		scalePanel.add( rescaleBtn );
		bottomPanel.add( scalePanel, BorderLayout.SOUTH );
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
			chart.getStyler().setYAxisMin( (Math.floor( ymin ) - 0.2 ));
			chart.getStyler().setYAxisMax( (Math.ceil( ymax ) + 0.2 ));
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
			frame.add( bottomPanel, BorderLayout.CENTER );
			frame.pack();
			frame.setPreferredSize( new Dimension(DEFAULT_WIDTH, DEFAULT_MAX_HEIGHT ));
			topPanel.setPreferredSize( new Dimension(DEFAULT_WIDTH, DEFAULT_MIN_HEIGHT ));
			topPanel.revalidate();
			topPanel.repaint();
			frame.revalidate();
			frame.repaint();
		} 

		chart.updateXYSeries( "Average Loss", epochData, averageLossData, null );
		chart.updateXYSeries( "Validation Loss", epochData, validationLossData, null );
		chartPanel.revalidate();
		chartPanel.repaint();

	}

	public void updateProgressText(String text) {
		message.setText(text);
		message.repaint();
	}
	
	public void updateProgress(int epoch, int step ) {
		if (progressBar.isIndeterminate())
		{
			progressBar.setIndeterminate( false );
			progressBar.setMinimum( 0 );
			progressBar.setMaximum( 10 );
		}
		int maxBareSize = 10; // 10unit for 100%
		int remainPercent = ( ( 100 * step ) / nEpochSteps ) / maxBareSize;
		progressBar.setValue(remainPercent);
		message.setText( "Epoch " + epoch + "/" + nEpochs + ", step " + step + "/" + nEpochSteps);
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
