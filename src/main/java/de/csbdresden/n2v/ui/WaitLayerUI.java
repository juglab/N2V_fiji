package de.csbdresden.n2v.ui;

import javax.swing.ImageIcon;
import javax.swing.JComponent;
import javax.swing.JLayer;
import javax.swing.JPanel;
import javax.swing.Timer;
import javax.swing.plaf.LayerUI;
import java.awt.AlphaComposite;
import java.awt.Color;
import java.awt.Composite;
import java.awt.Container;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.ImageObserver;
import java.beans.PropertyChangeEvent;
import java.net.URL;


public class WaitLayerUI extends LayerUI<Container> implements ActionListener, ImageObserver {

	private static final long serialVersionUID = 1L;
	private final JPanel parent;
	private boolean mIsRunning;
	private boolean mIsFadingOut;
	private Timer mTimer;

	private int mAngle;
	private int mFadeCount;
	private int mFadeLimit = 15;

	private ImageIcon waitingIcon;
	private Image waitingImage;
	private float iconScale;
	private int iconOffsetX, iconOffsetY;
	// top(0) right(1) bottom(2) left(3)
	private int iconAlignment;

	WaitLayerUI(JPanel parent){
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
		if(waitingImage != null) {
			//iconAlignment 2
			int iconX = w / 2 - Math.round(waitingIcon.getIconWidth() * iconScale / 2);
			int iconY = h - Math.round(waitingIcon.getIconHeight() * iconScale) -35;
			if(iconAlignment == 1) {
				iconX = w - Math.round(waitingIcon.getIconWidth() * iconScale);
				iconY = h / 2 - Math.round(waitingIcon.getIconHeight() * iconScale*0.5f);
			}
			g2.drawImage(waitingImage, iconX + iconOffsetX, iconY + iconOffsetY, this);
		}
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

	void stop() {
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

	void setWaitingIcon(URL iconUrl, float iconScale, int iconAlignment, int iconOffsetX, int iconOffsetY) {
		waitingIcon = new ImageIcon(iconUrl);
		this.iconScale = iconScale;
		this.iconOffsetX = iconOffsetX;
		this.iconOffsetY = iconOffsetY;
		this.iconAlignment = iconAlignment;
		waitingImage = waitingIcon.getImage().getScaledInstance(Math.round(iconScale*waitingIcon.getIconWidth()), Math.round(iconScale*waitingIcon.getIconHeight()), Image.SCALE_FAST);
	}
}
