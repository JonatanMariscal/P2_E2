import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import SimpleITK as sitk
import numpy as np
from skimage import measure


def resample_image(image, reference):
    pixel_spacing = image.GetSpacing()
    new_spacing = [old_sz * old_spc / new_sz for old_sz, old_spc, new_sz in
                   zip(image.GetSize(), pixel_spacing, reference.GetSize())]

    image_resampled = sitk.Resample(image, reference.GetSize(), sitk.Transform(), sitk.sitkNearestNeighbor,
                                    image.GetOrigin(), new_spacing,
                                    image.GetDirection(), 0.0, image.GetPixelIDValue())
    return image_resampled


# Register two images with same shape.
def register_images(image, reference):
    initial_transform = sitk.CenteredTransformInitializer(sitk.Cast(reference, image.GetPixelID()),
                                                          image,
                                                          sitk.Euler3DTransform(),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=250)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    registration_method.SetInterpolator(sitk.sitkNearestNeighbor)
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=10000,
                                                      convergenceMinimumValue=1e-6, convergenceWindowSize=10)

    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    final_transform = registration_method.Execute(sitk.Cast(reference, sitk.sitkFloat32),
                                                 sitk.Cast(image, sitk.sitkFloat32))
    register = sitk.ResampleImageFilter()
    register.SetReferenceImage(reference)
    register.SetInterpolator(sitk.sitkNearestNeighbor)
    register.SetTransform(final_transform)
    ds_register = register.Execute(image)

    return ds_register


def create_canvas(figure,frame):

    canvas = FigureCanvasTkAgg(figure, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, expand=1)

    toolbar = NavigationToolbar2Tk(canvas, frame)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, expand=1)


def main():

    def visualicer_selector():
        status = sel_visualicer.get()
        mode = selector.get()
        if not mode:
            if not status:
                frame_alpha.tkraise()
                sel_visualicer.set(True)
            else:
                frame.tkraise()
                sel_visualicer.set(False)

    def mode_selector():
        mode = selector.get()
        if not mode:
            frame_segmentation.tkraise()
            slice_selector.configure(to=ds_lung_array.shape[0]-1)
            selector.set(True)
        else:
            status = sel_visualicer.get()
            slice_selector.configure(to=image_1.shape[0]-1)
            if not status:
                frame.tkraise()
            else:
                frame_alpha.tkraise()
            selector.set(False)

    def update_slice(self):
        img_position = slice_selector.get()
        alpha_value = alpha_selector.get()
        status = sel_visualicer.get()
        mode = selector.get()
        if not mode:
            if not status:
                axs[0].imshow(image_1[img_position, :, :], cmap=plt.cm.get_cmap(colormap.get()))
                axs[1].imshow(image_2[img_position, :, :], cmap=plt.cm.get_cmap(colormap.get()))
                fig.canvas.draw_idle()
            else:
                ax.imshow(image_2[img_position, :, :], cmap=plt.cm.get_cmap("gray"))
                ax.imshow(image_1[img_position, :, :], cmap=plt.cm.get_cmap("Reds"), alpha=alpha_value/100)
                fig_alpha.canvas.draw_idle()
        else:
            axs_seg[0].imshow(ds_segmented_array[img_position, :, :], cmap=plt.cm.get_cmap("gray"))
            fig_seg.canvas.draw_idle()

    import time

    MAX_CLICK_LENGTH = 0.1  # in seconds; anything longer is a drag motion

    def onclick(event):
        axs_seg[0].time_onclick = time.time()

    def onrelease(event):
        if event.inaxes == axs_seg[0] and ((time.time() - axs_seg[0].time_onclick) < MAX_CLICK_LENGTH):
            img_position = slice_selector.get()
            position = tuple([img_position, int(event.ydata), int(event.xdata)])

            # # position = tuple([44, 304, 355])

            thresh = ds_segmented_array[position]
            mask = ds_segmented_array >= thresh
            labels = measure.label(mask)

            props = measure.regionprops(labels)
            for i in range(len(props)):
                blob = False
                coords = props[i].coords
                bbox = props[i].bbox
                if not (bbox[0] <= position[0] <= bbox[3] and bbox[1] <= position[1] <= bbox[4] and bbox[2] <= position[2] <= bbox[5]):
                    for j in range(len(coords)):
                        labels[tuple(coords[j])] = 0

                for j in range(len(coords)):
                    if tuple(coords[j]) == position:
                        blob = True
                if blob:
                    # coords.T[[1,2]] = coords.T[[2, 1]]
                    for j in range(len(coords)):
                        labels[tuple(coords[j])] = 0

            region = labels == labels[position]
            ds_masked = ds_segmented_array >= region

            axs_seg[0].imshow(ds_segmented_array[img_position, :, :], cmap=plt.cm.get_cmap("gray"))
            axs_seg[1].imshow(ds_masked[img_position, :, :], cmap=plt.cm.get_cmap("Reds"))
            fig_seg.canvas.draw_idle()

    # Reading RM_Brain_3D-SPGR DICOM
    path_dcm = "data/RM_Brain_3D-SPGR"
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(path_dcm)
    reader.SetFileNames(dicom_names)
    ds = reader.Execute()
    ds_array = sitk.GetArrayFromImage(ds)  # z, y, x

    # Reading CT_Lung DICOM
    path_dcm2 = "data/CT_Lung"
    dicom_names2 = reader.GetGDCMSeriesFileNames(path_dcm2)
    reader.SetFileNames(dicom_names2)
    ds_lung = reader.Execute()
    ds_lung_array = sitk.GetArrayFromImage(ds_lung)  # z, y, x

    # Reading phantom DICOM
    ds_phantom = sitk.ReadImage('data/icbm_avg_152_t1_tal_nlin_symmetric_VI.dcm')
    phantom_array = sitk.GetArrayFromImage(ds_phantom)  # z, y, x

    # Reading atlas DICOM
    ds_atlas = sitk.ReadImage('data/AAL3_1mm.dcm')

    # Resample Brain DICOM and atlas DICOM to phantom shape
    ds_resample = resample_image(ds, ds_phantom)
    ds_atlas_resample = resample_image(ds_atlas, ds_phantom)

    # Register Brain DICOM and atlas DICOM with phantom
    ds_atlas_register = register_images(ds_atlas_resample, ds_phantom)

    atlas_array = sitk.GetArrayFromImage(ds_atlas_register)  # z, y, x
    atlas_array[atlas_array < 121] = 0
    atlas_array[atlas_array > 150] = 0

    ds_register = register_images(ds_resample, ds_phantom)
    ds_array_register = sitk.GetArrayFromImage(ds_register)  # z, y, x

    # Register atlas to brain original
    ds_atlas_res2 = resample_image(ds_atlas, ds)
    ds_atlas_reg2 = register_images(ds_atlas_res2, ds)
    atlas_array_brain = sitk.GetArrayFromImage(ds_atlas_reg2)  # z, y, x
    atlas_array_brain[atlas_array_brain < 121] = 0
    atlas_array_brain[atlas_array_brain > 150] = 0

    # Creating window and frames
    root = tk.Tk()
    root.title("DICOM Image Display")
    top_frame = tk.Frame() # frame with buttons and sliders
    frame = tk.Frame() #frame with synchron visualicer
    frame_alpha = tk.Frame() #frame with alpha visualicer
    frame_segmentation = tk.Frame() #frame with segmentation visualicer

    top_frame.grid(row = 0, column = 0, sticky = tk.W, columnspan=6)
    frame.grid(row = 1,sticky="nsew", column = 0, columnspan=6)
    frame_alpha.grid(row = 1,sticky="nsew", column = 0, columnspan=6)
    frame_segmentation.grid(row = 1,sticky="nsew", column = 0, columnspan=6)
    frame.tkraise()

    sel_visualicer = tk.BooleanVar()
    selector = tk.BooleanVar()

    # Select which images to display:
    # Brain -> ds_array / Brain_register -> ds_array_register / phantom -> phantom_array / atlas_register -> atlas_array
    # atlas register to brain -> atlas_array_brain
    image_1 = ds_array_register
    image_2 = phantom_array

    # Displaying images on synchron visualizer
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), dpi=100, sharex=True, sharey=True)
    axs = axs.ravel()
    colormap = tk.StringVar()
    colormap.set("bone")

    axs[0].imshow(image_1[0,:,:], cmap=plt.cm.get_cmap(colormap.get()))
    axs[1].imshow(image_2[0,:,:], cmap=plt.cm.get_cmap(colormap.get()))

    create_canvas(fig, frame)

    # Displaying images on alpha visualizer
    fig_alpha, ax = plt.subplots(1, figsize=(12, 4), dpi=100, sharex=True, sharey=True)

    alpha = 0
    ax.imshow(image_2[0, :, :], cmap=plt.cm.get_cmap("gray"))
    ax.imshow(image_1[0, :, :], cmap=plt.cm.get_cmap("Reds"), alpha=alpha/100)

    create_canvas(fig_alpha, frame_alpha)

    # Displaying image on segment visualizer
    fig_seg, axs_seg = plt.subplots(1, 2, figsize=(12, 4), dpi=100, sharex=True, sharey=True)
    axs_seg = axs_seg.ravel()

    feature_img = sitk.GradientMagnitude(ds_lung)

    ds_segmented_array = np.flip(sitk.GetArrayFromImage(ds_lung), axis=0)
    axs_seg[0].imshow(ds_segmented_array[0, :, :], cmap=plt.cm.get_cmap("gray"))
    create_canvas(fig_seg, frame_segmentation)

    # Selecting slices
    pos = 0
    slice_selector = tk.Scale(top_frame, label="Slice selector", from_=0, to=image_1.shape[0] - 1,
                              orient=tk.HORIZONTAL, length=400,
                              command=update_slice, tickinterval=20)
    slice_selector.grid(rowspan=2, column=0)
    # Change between Visualization and segmentation
    button_visual_seg = tk.Button(top_frame, wraplength=80, text="Visualizer mode selector", command=visualicer_selector, width=10)
    button_visual_seg.grid(row=0, column=2)

    # Change between synchron and alhpa visualization
    button_syn_alpha = tk.Button(top_frame, wraplength=80, text="Mode selector", command=mode_selector, width=10)
    button_syn_alpha.grid(row=1, column=2)

    # Selecting which percentage of alpha use for alpha visualization
    alpha_selector = tk.Scale(top_frame, label="Alpha value (%)", from_=0, to=100,
                              orient=tk.HORIZONTAL, length=400,
                              command=update_slice, tickinterval=5)
    alpha_selector.grid(rowspan=2, column=0)

    # Segmentation process: Selecting marker for watershed segmentation

    fig_seg.canvas.mpl_connect('button_press_event', onclick)
    fig_seg.canvas.mpl_connect('button_release_event', onrelease)

    root.mainloop()


if __name__ == '__main__':
    main()
