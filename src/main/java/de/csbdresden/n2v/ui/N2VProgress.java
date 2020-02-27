package de.csbdresden.n2v.ui;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Font;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.List;

import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.SwingConstants;
import javax.swing.text.SimpleAttributeSet;

import org.scijava.app.StatusService;
import org.scijava.thread.ThreadService;

import de.csbdresden.n2v.train.N2VTraining;

public class N2VProgress extends JPanel {

	private static final long serialVersionUID = 1L;
	private class GuiTask {


		public JLabel status;
		public JLabel title;
		public JLabel progress;
		public boolean taskDone;
	}

	private final static String FRAME_TITLE = "N2V for Fiji";
	private final static ImageIcon busyIcon = new ImageIcon( N2VProgress.class.getClassLoader().getResource( "ajax-loader.gif" ) );

	//private final StatusService status;
	private final ThreadService threadService;

	public static final int STATUS_IDLE = -1;
	public static final int STATUS_RUNNING = 0;
	public static final int STATUS_DONE = 1;
	public static final int STATUS_FAIL = 2;

	private final List< GuiTask > tasks = new ArrayList<>();
	private int currentTask;

	private final JPanel taskContainer;
	private final JFrame frame;
	private final JButton finishBtn;
	private final JButton cancelBtn;

	private final SimpleAttributeSet red = new SimpleAttributeSet();

	private N2VTraining n2v;
	private N2VChartPanel chart;
	private JLabel warningLabel;

	public N2VProgress( JFrame frame, N2VTraining n2v, int nEpochs, int nEpochSteps, StatusService status, ThreadService threadService ) {

		super( new BorderLayout() );
		setBackground( Color.WHITE );
		this.n2v = n2v;

		//this.status = status;
		this.frame = frame;
		this.threadService = threadService;

		final JPanel warnrow = new JPanel( new FlowLayout( FlowLayout.LEFT ) );
		warnrow.setBackground(Color.WHITE);
		warningLabel = new JLabel("",JLabel.CENTER);
		warningLabel.setPreferredSize( new Dimension(500, 10));
		warningLabel.setForeground( Color.RED );
		warnrow.add( warningLabel );
		
		taskContainer = new JPanel();
		taskContainer.setLayout( new BoxLayout( taskContainer, BoxLayout.Y_AXIS ) );
		taskContainer.setBackground( Color.WHITE );
		taskContainer.add(warnrow);

		add( taskContainer, BorderLayout.PAGE_START );

		final JPanel centerPanel = new JPanel();
		centerPanel.setBackground( Color.WHITE );
		centerPanel.setLayout( new BoxLayout( centerPanel, BoxLayout.Y_AXIS ) );
		centerPanel.add( Box.createRigidArea( new Dimension( 10, 0 ) ) );
		chart = new N2VChartPanel( nEpochs, nEpochSteps );
		centerPanel.add( chart.getPanel() );
		add( centerPanel, BorderLayout.CENTER );

		//resetProgress();

		// Buttons panel
		JPanel buttonsPanel = new JPanel();
		buttonsPanel.setLayout(new GridBagLayout());
		buttonsPanel.setBackground( Color.WHITE );
		GridBagConstraints gbc = new GridBagConstraints();
		gbc.anchor = GridBagConstraints.CENTER;
		gbc.fill = GridBagConstraints.NONE;
		gbc.insets = new Insets( 5, 5, 5, 5 );
		gbc.gridx = 0;

		cancelBtn = new JButton( "Cancel" );
		cancelBtn.addActionListener(e -> n2v.cancel());
		buttonsPanel.add( cancelBtn, gbc );

		gbc.gridx = 1;
		finishBtn = new JButton( "Finish Training" );
		finishBtn.addActionListener( e -> n2v.stopTraining() );
		finishBtn.setEnabled(false);
		buttonsPanel.add( finishBtn, gbc );

		add( buttonsPanel, BorderLayout.SOUTH );

		invalidate();
		repaint();

		frame.setContentPane( this );
		frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);

	}

	public void display() {
		// Display the window.
		try {
			threadService.invoke( () -> {
				frame.pack();
				frame.setLocationRelativeTo( null );
				frame.setVisible( true );
			} );
		} catch ( InterruptedException | InvocationTargetException e ) {
			e.printStackTrace();
		}
	}

	public void addTask( final String title ) {
		final JPanel taskrow = new JPanel( new FlowLayout( FlowLayout.LEFT ) );
		taskrow.setBackground(Color.WHITE);
		final JLabel statusLabel = new JLabel( "\u2013", SwingConstants.CENTER );
		final Font font = statusLabel.getFont();
		statusLabel.setFont( new Font( font.getName(), Font.BOLD, font.getSize() * 2 ) );
		statusLabel.setPreferredSize( new Dimension( 50, 30 ) );
		statusLabel.setMinimumSize( new Dimension( 50, 30 ) );
		statusLabel.setMaximumSize( new Dimension( 50, 30 ) );
		final GuiTask task = new GuiTask();
		task.status = statusLabel;
		task.title = new JLabel( title );
		JLabel progressLabel = new JLabel( "" );
		progressLabel.setForeground( new Color( 32, 32, 32 ) );
		task.progress = progressLabel;
		task.taskDone = false;
		tasks.add( task );
		taskrow.add( task.status );
		taskrow.add( task.title );
		taskrow.add( task.progress );
		taskContainer.add( taskrow );
	}

	public void setTaskStart( final int task ) {
		if(task == 1) {
			finishBtn.setEnabled(true);
		}
		currentTask = task;
		setCurrentTaskStatus( STATUS_RUNNING );
	}

	public void setTaskDone( final int task ) {
		if(task == 2) {
			cancelBtn.setEnabled(false);
			finishBtn.setText("Close");
		}
		tasks.get( currentTask ).taskDone = true;
		setTaskStatus( task, STATUS_DONE );
		setTaskMessage( task, "" );
		//status.clearStatus();
	}

	public void setTaskFail( final int task ) {
		setTaskStatus( task, STATUS_FAIL );
		//status.clearStatus();
	}

	public void setTaskNumSteps( final int task, final int steps ) {
		//tasks.get( currentTask ).numSteps = steps;
	}

	public void setTaskCurrentStep( final int task, final int step ) {
		//tasks.get( currentTask ).step = step;
	}

	public void setCurrentTaskStatus( final int status ) {
		setTaskStatus( currentTask, status );
	}

	public void setCurrentTaskMessage( final String text ) {
		setTaskMessage( currentTask, text );
	}

	private void setTaskMessage( final int task, String text ) {
		if(!text.isEmpty()) text = "- " + text;
		tasks.get( task ).progress.setText( text );
	}

	private void setTaskStatus( final int task, final int status ) {

		//if ( status < tasks.size() && task >= 0 ) {
		final JLabel statuslabel = tasks.get( task ).status;
		switch ( status ) {
		case STATUS_IDLE:
			statuslabel.setIcon( null );
			statuslabel.setText( "\u2013" );
			statuslabel.setForeground( Color.getHSBColor( 0.6f, 0.f, 0.3f ) );
			break;
		case STATUS_RUNNING:
			statuslabel.setIcon( busyIcon );
			statuslabel.setText( "" );
			statuslabel.setForeground( Color.getHSBColor( 0.6f, 0.f, 0.3f ) );
			break;
		case STATUS_DONE:
			statuslabel.setIcon( null );
			statuslabel.setText( "\u2713" );
			statuslabel.setForeground( Color.getHSBColor( 0.3f, 1, 0.6f ) );
			break;
		case STATUS_FAIL:
			statuslabel.setIcon( null );
			statuslabel.setText( "\u2717" );
			statuslabel.setForeground( Color.RED );
			break;
		}
		//}
	}

	public static N2VProgress create( N2VTraining n2v, int nEpochs, int nEpochSteps, StatusService status, ThreadService threadService ) {

		// Create and set up the window.
		final JFrame frame = new JFrame( FRAME_TITLE );

		// Create and set up the content pane.
		final N2VProgress newContentPane = new N2VProgress( frame, n2v, nEpochs, nEpochSteps, status, threadService );

		return newContentPane;
	}

	public void dispose() {
		if ( frame.isVisible() ) {
			frame.dispose();
		}
	}

	public void updateTrainingChart( int i, List< Double > losses, float validationLoss ) {
		chart.updateChart( i, losses, validationLoss );
	}

	public void updateTrainingProgress( int i, int j ) {
		chart.updateProgress( i, j );
	}

	public void setWarning( String string ) {
		warningLabel.setText( string );
	}
}
