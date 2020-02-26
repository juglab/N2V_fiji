package de.csbdresden.n2v.ui;

import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.ImageObserver;
import java.beans.PropertyChangeEvent;

import javax.swing.*;
import javax.swing.plaf.LayerUI;


public class WaitLayerUI extends LayerUI<Container> implements ActionListener, ImageObserver {

	private static final long serialVersionUID = 1L;
	private final JPanel parent;
	private boolean mIsRunning;
	private boolean mIsFadingOut;
	private Timer mTimer;

	private int mAngle;
	private int mFadeCount;
	private int mFadeLimit = 15;

	private final static ImageIcon waitingIcon = new ImageIcon( N2VProgress.class.getClassLoader().getResource( "hard-workout.gif" ) );
	private final static int iconScale = 2;
	private Image waitingImage = waitingIcon.getImage().getScaledInstance(iconScale*waitingIcon.getIconWidth(), iconScale*waitingIcon.getIconHeight(), Image.SCALE_FAST);

	public WaitLayerUI(JPanel parent){
		this.parent = parent;
	}

	@Override
	public void paint(Graphics g, JComponent c) {
		int w = c.getWidth();
		int h = c.getHeight();

		// Paint the view.
		super.paint(g, c);

		if (!mIsRunning) {
			return;
		}

		Graphics2D g2 = (Graphics2D) g.create();

		float fade = (float) mFadeCount / (float) mFadeLimit;
		// Gray it out.
		Composite urComposite = g2.getComposite();
		g2.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, .7f * fade));
		g2.setPaint(Color.white);
		g2.fillRect(0, 0, w, h);
		g2.drawImage(waitingImage, w/2-waitingIcon.getIconWidth()*iconScale/2, h - waitingIcon.getIconHeight()*iconScale - 35, this);
		g2.setComposite(urComposite);

		g2.dispose();
	}

	@Override
  public void actionPerformed(ActionEvent e) {
		if (mIsRunning) {
			firePropertyChange("tick", 0, 1);
			mAngle += 3;
			if (mAngle >= 360) {
				mAngle = 0;
			}
			if (mIsFadingOut) {
				if (--mFadeCount == 0) {
					mIsRunning = false;
					mTimer.stop();
				} else {
					--mFadeCount;
				}
			} else if (mFadeCount < mFadeLimit) {
				mFadeCount++;
			}
		}
	}

	public void start() {
		// Run a thread for animation.
		mIsRunning = true;
		mIsFadingOut = false;
		mFadeCount = 0;
		int fps = 24;
		int tick = 1000 / fps;
		mTimer = new Timer(tick, this);
		mTimer.start();
	}

	public void stop() {
		mIsFadingOut = true;
	}

	@Override
	public void applyPropertyChange(PropertyChangeEvent pce, @SuppressWarnings("rawtypes") JLayer l) {
		if ("tick".equals(pce.getPropertyName())) {
			l.repaint();
		}
	}

	@Override
	public boolean imageUpdate(Image image, int i, int i1, int i2, int i3, int i4) {
		parent.repaint();
		return true;
	}
}
