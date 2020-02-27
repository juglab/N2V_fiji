package de.csbdresden.n2v;

import net.imagej.ops.OpService;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;
import org.scijava.Context;
import org.scijava.display.Display;
import org.scijava.display.DisplayService;
import org.scijava.plugin.Parameter;
import org.scijava.ui.UIService;

import java.util.ArrayList;
import java.util.List;

public class PreviewHandler {

	@Parameter
	private UIService uiService;

	@Parameter
	private OpService opService;

	private final int trainDimensions;
	private RandomAccessibleInterval<FloatType> splitImage;
	private List<RandomAccessibleInterval<FloatType>> historyImages;

	public PreviewHandler(Context context, int trainDimensions) {
		context.inject(this);
		this.trainDimensions = trainDimensions;
	}

	public void update(RandomAccessibleInterval<FloatType> in, RandomAccessibleInterval<FloatType> out) {
		updateSplitImage(in, out);
//		updateHistoryImage(out);
	}

	private void updateSplitImage(RandomAccessibleInterval<FloatType> in, RandomAccessibleInterval<FloatType> out) {
		if (Thread.interrupted()) return;
		if(splitImage == null) splitImage = opService.copy().rai(out);
		else opService.copy().rai(splitImage, out);
		if(trainDimensions == 2) updateSplitImage2D(in);
		if(trainDimensions == 3) updateSplitImage3D(in);
		Display<?> display = uiService.context().service(DisplayService.class).getDisplay("training preview");
		if(display == null) uiService.show("training preview", splitImage);
		else display.update();
	}

	private void updateSplitImage2D(RandomAccessibleInterval<FloatType> in) {
		RandomAccess<FloatType> inRA = in.randomAccess();
		RandomAccess<FloatType> splitRA = splitImage.randomAccess();
		for (int i = 0; i < in.dimension(0); i++) {
			for (int j = 0; j < in.dimension(1); j++) {
				if(i < in.dimension(1)-j) {
					inRA.setPosition(i, 0);
					inRA.setPosition(j, 1);
					for (int k = 0; k < in.dimension(2); k++) {
						inRA.setPosition(k, 2);
						splitRA.setPosition(inRA);
						splitRA.get().set(inRA.get());
					}
				}
			}
		}
	}

	private void updateSplitImage3D(RandomAccessibleInterval<FloatType> in) {
		RandomAccess<FloatType> inRA = in.randomAccess();
		RandomAccess<FloatType> splitRA = splitImage.randomAccess();
		for (int i = 0; i < in.dimension(0); i++) {
			inRA.setPosition(i, 0);
			for (int j = 0; j < in.dimension(1); j++) {
				if(i < in.dimension(1)-j) {
					inRA.setPosition(j, 1);
					for (int k = 0; k < in.dimension(2); k++) {
						inRA.setPosition(k, 2);
						for (int l = 0; l < in.dimension(3); l++) {
							inRA.setPosition(l, 3);
							splitRA.setPosition(inRA);
							splitRA.get().set(inRA.get());
						}
					}
				}
			}
		}
	}

	private void updateHistoryImage(RandomAccessibleInterval<FloatType> out) {
		for (int i = 2; i < out.numDimensions(); i++) {
			out = Views.hyperSlice(out, i, 0);
		}
		//TODO copying neccessary?
		RandomAccessibleInterval<FloatType> outXY = opService.copy().rai(out);
		if(historyImages == null) historyImages = new ArrayList<>();
		historyImages.add(0, outXY);
		Display<?> display = uiService.context().service(DisplayService.class).getDisplay("training history");
		RandomAccessibleInterval<FloatType> stack = Views.stack(historyImages);
		if(display == null) uiService.show("training history", stack);
		else {
			display.clear();
			display.display(stack);
			display.update();
		}
	}

}
