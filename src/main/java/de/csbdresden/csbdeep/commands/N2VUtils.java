package de.csbdresden.csbdeep.commands;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;
import org.apache.commons.lang3.NotImplementedException;

import java.util.List;


//manipulator:
//def pm_uniform_withCP(local_sub_patch_radius):
//    def random_neighbor_withCP_uniform(patch, coords, dims):
//        vals = []
//        for coord in zip(*coords):
//            sub_patch = get_subpatch(patch, coord,local_sub_patch_radius)
//            rand_coords = [np.random.randint(0, s) for s in sub_patch.shape[0:dims]]
//            vals.append(sub_patch[tuple(rand_coords)])
//        return vals
//    return random_neighbor_withCP_uniform

public class N2VUtils {
	private static int n2v_neighborhood_radius = 5;
	public static void manipulate_val_data(RandomAccessibleInterval<FloatType> X_val, RandomAccessibleInterval Y_val, double perc_pix, List<Long> shape) {
		int dims = shape.size();
		long box_size;
		if(dims == 2) {
			box_size = Math.round(Math.sqrt(100./perc_pix));
		} else {
			throw new NotImplementedException("manipulate_val_data not implemented for dim>2");
		}
		long n_chan = X_val.dimension(X_val.numDimensions()-1);
		Views.iterable(X_val).forEach(val -> val.set(0));
		for (int j = 0; j < X_val.dimension(dims); j++) {
			N2V_DataWrapper.manipulateY(j, box_size, shape, X_val, Y_val, dims, n_chan);
		}
	}

//	def manipulate_val_data(X_val, Y_val, perc_pix=0.198, shape=(64, 64), value_manipulation=pm_uniform_withCP(5)):
//    dims = len(shape)
//    if dims == 2:
//        box_size = np.round(np.sqrt(100/perc_pix)).astype(np.int)
//        get_stratified_coords = dw.__get_stratified_coords2D__
//        rand_float = dw.__rand_float_coords2D__(box_size)
//    elif dims == 3:
//        box_size = np.round(np.sqrt(100/perc_pix)).astype(np.int)
//        get_stratified_coords = dw.__get_stratified_coords3D__
//        rand_float = dw.__rand_float_coords3D__(box_size)
//
//    n_chan = X_val.shape[-1]
//
//    Y_val *= 0
//    for j in tqdm(range(X_val.shape[0]), desc='Preparing validation data: '):
//        coords = get_stratified_coords(rand_float, box_size=box_size,
//                                            shape=np.array(X_val.shape)[1:-1])
//        for c in range(n_chan):
//            indexing = (j,) + coords + (c,)
//            indexing_mask = (j,) + coords + (c + n_chan,)
//            y_val = X_val[indexing]
//            x_val = value_manipulation(X_val[j, ..., c], coords, dims)
//
//            Y_val[indexing] = y_val
//            Y_val[indexing_mask] = 1
//            X_val[indexing] = x_val

	private static float[] rand_float_coords2D(long boxsize) {
		return new float[]{(float) (Math.random() * boxsize), (float) (Math.random() * boxsize)};
	}

}
