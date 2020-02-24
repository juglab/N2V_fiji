package de.csbdresden.n2v.ui;

import java.awt.Color;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;

import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JProgressBar;
import javax.swing.SwingConstants;

public class TrainingProgressPanel extends JPanel {

	private static final long serialVersionUID = 1L;

	private JProgressBar epochProgressBar;
	private JProgressBar stepProgressBar;
	private int nEpochs;
	private int nEpochSteps;

	private JLabel epochProgressLabel;

	private JLabel stepProgressLabel;

	public TrainingProgressPanel( int nEpochs, int nEpochSteps ) {
		this.nEpochs = nEpochs;
		this.nEpochSteps = nEpochSteps;
		setLayout( new GridBagLayout() );
		setBackground( Color.WHITE);
		GridBagConstraints gbc = new GridBagConstraints();
		gbc.fill = GridBagConstraints.BOTH;
		gbc.anchor = GridBagConstraints.WEST;
		gbc.insets = new Insets( 5, 10, 2, 5 );
		gbc.gridy = 0;
		epochProgressLabel = new JLabel("Epoch", JLabel.LEFT);
		add( epochProgressLabel, gbc );
		
		//For a thicker progress bar
		//UIManager.put( "ProgressBarUI", "javax.swing.plaf.metal.MetalProgressBarUI" );
		gbc.gridwidth = GridBagConstraints.RELATIVE;
		gbc.weightx = 1;
		gbc.insets = new Insets( 5, 5, 2, 10 );
		epochProgressBar = new JProgressBar( SwingConstants.HORIZONTAL );
		epochProgressBar.setMinimum( 0 );
		epochProgressBar.setMaximum( nEpochs );
		add( epochProgressBar, gbc );

		gbc.anchor = GridBagConstraints.WEST;
		gbc.gridy = 1;
		gbc.weightx = 0;
		gbc.insets = new Insets( 2, 10, 5, 5 );
		stepProgressLabel = new JLabel("Step", JLabel.LEFT);
		add( stepProgressLabel, gbc );
		
		gbc.gridwidth = GridBagConstraints.RELATIVE;
		gbc.insets = new Insets( 2, 5, 5, 10 );
		gbc.weightx = 1;
		stepProgressBar = new JProgressBar( SwingConstants.HORIZONTAL );
		stepProgressBar.setMinimum( 0 );
		stepProgressBar.setMaximum( nEpochSteps );
		add( stepProgressBar, gbc );

	}

	public static void main( String[] args ) {
		final JFrame frame = new JFrame();
		TrainingProgressPanel panel = new TrainingProgressPanel( 5, 5 );
		frame.getContentPane().add( panel );
		frame.pack();
		frame.setLocationRelativeTo( null );
		frame.setVisible( true );
	}

	public void updateProgress( int epoch, int step ) {

		if ( step == 1 ) {
			epochProgressBar.setValue( epoch );
			epochProgressLabel.setText( "Epoch " + epoch + "/" + nEpochs );
		}
		stepProgressBar.setValue( step );
		stepProgressLabel.setText( "Step " + step + "/" + nEpochSteps );
		repaint();
	}
}
