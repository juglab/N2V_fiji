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
