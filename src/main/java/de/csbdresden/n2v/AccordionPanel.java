package de.csbdresden.n2v;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Component;
import java.awt.Dimension;
import java.awt.GradientPaint;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Point;
import java.awt.RenderingHints;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.util.List;

import javax.swing.BorderFactory;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;

import de.csbdresden.n2v.N2VDialog.TrainingStepStatus;
import de.csbdresden.n2v.N2VDialog.TrainingSteps;

public class AccordionPanel extends JPanel {

	boolean movingComponents = false;
	int visibleIndex;

	public AccordionPanel( List< String > titles ) {
		setLayout( null );
		// Add children and compute prefSize.
		int childCount = titles.size();
		visibleIndex = childCount - 1;
		Dimension d = new Dimension();
		int h = 0;
		for ( int j = 0; j < childCount; j++ ) {
			ChildPanel child = new ChildPanel( j + 1, titles.get( j ), ml );
			add( child );
			d = child.getPreferredSize();
			child.setBounds( 0, h, d.width, d.height );
			if ( j < childCount - 1 )
				h += HeaderPanel.HEIGHT;
		}
		h += d.height;
		setPreferredSize( new Dimension( d.width, h ) );
		// Set z-order for children.
		setZOrder();
	}

	private void setZOrder() {
		Component[] c = getComponents();
		for ( int j = 0; j < c.length - 1; j++ ) {
			setComponentZOrder( c[ j ], c.length - 1 - j );
		}
	}

	public void setChildVisible( int indexToOpen ) {
		// If visibleIndex < indexToOpen, components at
		// [visibleIndex+1 down to indexToOpen] move up.
		// If visibleIndex > indexToOpen, components at
		// [indexToOpen+1 up to visibleIndex] move down.
		// Collect indices of components that will move
		// and determine the distance/direction to move.
		int[] indices = new int[ 0 ];
		int travelLimit = 0;
		if ( visibleIndex < indexToOpen ) {
			travelLimit = HeaderPanel.HEIGHT - getComponent( visibleIndex ).getHeight();
			int n = indexToOpen - visibleIndex;
			indices = new int[ n ];
			for ( int j = visibleIndex, k = 0; j < indexToOpen; j++, k++ )
				indices[ k ] = j + 1;
		} else if ( visibleIndex > indexToOpen ) {
			travelLimit = getComponent( visibleIndex ).getHeight() - HeaderPanel.HEIGHT;
			int n = visibleIndex - indexToOpen;
			indices = new int[ n ];
			for ( int j = indexToOpen, k = 0; j < visibleIndex; j++, k++ )
				indices[ k ] = j + 1;
		}
		movePanels( indices, travelLimit );
		visibleIndex = indexToOpen;
	}

	private void movePanels( final int[] indices, final int travel ) {
		movingComponents = true;
		Thread thread = new Thread( new Runnable() {

			public void run() {
				Component[] c = getComponents();
				int limit = travel > 0 ? travel : 0;
				int count = travel > 0 ? 0 : travel;
				int dy = travel > 0 ? 1 : -1;
				while ( count < limit ) {
					try {
						Thread.sleep( 25 );
					} catch ( InterruptedException e ) {
						System.out.println( "interrupted" );
						break;
					}
					for ( int j = 0; j < indices.length; j++ ) {
						// The z-order reversed the order returned
						// by getComponents. Adjust the indices to
						// get the correct components to relocate.
						int index = c.length - 1 - indices[ j ];
						Point p = c[ index ].getLocation();
						p.y += dy;
						c[ index ].setLocation( p.x, p.y );
					}
					repaint();
					count++;
				}
				movingComponents = false;
			}
		} );
		thread.setPriority( Thread.NORM_PRIORITY );
		thread.start();
	}

	private MouseListener ml = new MouseAdapter() {

		public void mousePressed( MouseEvent e ) {
			int index = ( ( HeaderPanel ) e.getSource() ).id - 1;
			if ( !movingComponents )
				setChildVisible( index );
		}
	};

	public void setChildStatus( int id, TrainingStepStatus status ) {
		( ( ChildPanel ) getComponent( id ) ).setStatus( status );
	}

	void addChildPanelContent( int id, JPanel panel ) {
		( ( ChildPanel ) getComponent( id ) ).addContent( panel );
	}

	public JPanel getPanel() {
		JPanel panel = new JPanel( new GridBagLayout() );
		GridBagConstraints gbc = new GridBagConstraints();
		panel.setBorder( BorderFactory.createLineBorder( Color.black, 1 ) );
		panel.add( this, gbc );
		return panel;
	}

	public static void main( String[] args ) {
		JFrame f = new JFrame();
		f.setDefaultCloseOperation( JFrame.EXIT_ON_CLOSE );
		JPanel panel = new JPanel( new GridBagLayout() );
		GridBagConstraints gbc = new GridBagConstraints();
		panel.add( new AccordionPanel( TrainingSteps.getStepNames() ).getPanel(), gbc );
		f.getContentPane().add( panel );
		f.setSize( 400, 400 );
		f.setLocation( 200, 200 );
		f.setVisible( true );
	}

}

class ChildPanel extends JPanel {

	private HeaderPanel header;

	public ChildPanel( int id, String title, MouseListener ml ) {
		setLayout( new BorderLayout() );
		header = new HeaderPanel( id, title, ml );
		add( header, "First" );
	}

	public void addContent( JPanel contentPanel ) {
		add( contentPanel );
	}

	public void setStatus( TrainingStepStatus status ) {
		header.setStatus( status );
	}

	public Dimension getPreferredSize() {
		return new Dimension( 300, 150 );
	}
}

class HeaderPanel extends JPanel {

	int id;
	String title;
	JLabel titleLabel;
	Color c1 = new Color( 200, 180, 180 );
	Color c2 = new Color( 200, 220, 220 );
	Color fontFg = Color.blue;
	Color rolloverFg = Color.red;
	public final static int HEIGHT = 45;

	public HeaderPanel( int id, String title, MouseListener ml ) {
		this.id = id;
		this.title = title;
		setLayout( new BorderLayout() );
		add( titleLabel = new JLabel( TrainingStepStatus.IDLE.getStatusIcon() + " " + title, JLabel.CENTER ) );
		titleLabel.setForeground( fontFg );
		Dimension d = getPreferredSize();
		d.height = HEIGHT;
		setPreferredSize( d );
		addMouseListener( ml ); //this opens or closes if mouse over
	}

	protected void paintComponent( Graphics g ) {
		int w = getWidth();
		Graphics2D g2 = ( Graphics2D ) g;
		g2.setRenderingHint(
				RenderingHints.KEY_ANTIALIASING,
				RenderingHints.VALUE_ANTIALIAS_ON );
		g2.setPaint( new GradientPaint( w / 2, 0, c1, w / 2, HEIGHT / 2, c2 ) );
		g2.fillRect( 0, 0, w, HEIGHT );
	}

	public void setStatus( TrainingStepStatus status ) {
		titleLabel.setText( status.getStatusIcon() + " " + title );
		repaint();
	}

}
