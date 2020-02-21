package de.csbdresden.n2v.ui;

import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;

import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JProgressBar;
import javax.swing.SwingConstants;

public class TrainingProgressPanel extends JPanel {

	private static final long serialVersionUID = 1L;

	private JProgressBar epochProgressBar;
	private JProgressBar stepProgressBar;
	private int nEpochs;
	private int nEpochSteps;

	public TrainingProgressPanel( int nEpochs, int nEpochSteps ) {
		this.nEpochs = nEpochs;
		this.nEpochSteps = nEpochSteps;
		setLayout( new GridBagLayout() );
		GridBagConstraints gbc = new GridBagConstraints();
		gbc.anchor = GridBagConstraints.WEST;
		gbc.fill = GridBagConstraints.HORIZONTAL;
		gbc.insets = new Insets( 5, 2, 5, 2 );
		gbc.gridx = 0;
		gbc.gridy = 0;
		
		//For a thicker progress bar
		//UIManager.put( "ProgressBarUI", "javax.swing.plaf.metal.MetalProgressBarUI" );
		epochProgressBar = new JProgressBar( SwingConstants.HORIZONTAL );
		epochProgressBar.setStringPainted( true );
		epochProgressBar.setMinimum( 0 );
		epochProgressBar.setMaximum( nEpochs );
		add( epochProgressBar, gbc );

		gbc.gridy = 1;
		stepProgressBar = new JProgressBar( SwingConstants.HORIZONTAL );
		stepProgressBar.setStringPainted( true );
		stepProgressBar.setMinimum( 0 );
		stepProgressBar.setMaximum( nEpochSteps );
		add( stepProgressBar, gbc );

	}

	public static void main( String[] args ) {
		final JFrame frame = new JFrame();
		TrainingProgressPanel panel = new TrainingProgressPanel( 5, 5 );
		frame.add( panel );
		frame.pack();
		frame.setLocationRelativeTo( null );
		frame.setVisible( true );
	}

	public void updateProgress( int epoch, int step ) {

		if ( step == 1 ) {
			epochProgressBar.setValue( epoch );
			epochProgressBar.setString( "Epoch " + epoch + "/" + nEpochs );
		}
		stepProgressBar.setValue( step );
		stepProgressBar.setString( "Step " + step + "/" + nEpochSteps );
	}
}
