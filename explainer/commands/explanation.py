import shap

class Explanation:
    labels = {
        "MAIN_EFFECT": "SHAP main effect value for\n%s",
        "INTERACTION_VALUE": "SHAP interaction value",
        "INTERACTION_EFFECT": "SHAP interaction value for\n%s and %s",
        "VALUE": "SHAP value (impact on model output)",
        "GLOBAL_VALUE": "mean(|SHAP value|) (average impact on model output magnitude)",
        "VALUE_FOR": "SHAP value for\n%s",
        "PLOT_FOR": "SHAP plot for %s",
        "FEATURE": "Feature %s",
        "FEATURE_VALUE": "Feature value",
        "FEATURE_VALUE_LOW": "Low",
        "FEATURE_VALUE_HIGH": "High",
        "JOINT_VALUE": "Joint SHAP value",
        "MODEL_OUTPUT": "Model output value",
    }

    def __init__(self, shap_values: shap.Explanation, max_display=10, show=True):
        base_values = shap_values.base_values
        features = shap_values.data
        feature_names = shap_values.feature_names
        lower_bounds = getattr(shap_values, "lower_bounds", None)
        upper_bounds = getattr(shap_values, "upper_bounds", None)
        values = shap_values.values
        num_features = min(max_display, len(values))
        pos_lefts = []
        pos_inds = []
        pos_widths = []
        pos_low = []
        pos_high = []
        neg_lefts = []
        neg_inds = []
        neg_widths = []
        neg_low = []
        neg_high = []
        output_pts = []
        order = np.argsort(-np.abs(values))
        loc = base_values + values.sum()
        yticklabels = ["" for i in range(num_features + 1)]

        # make sure we only have a single output to explain
        if (type(base_values) == np.ndarray and len(base_values) > 0) or type(
            base_values
        ) == list:
            raise Exception(
                "waterfall_plot requires a scalar base_values of the model output as the first "
                "parameter, but you have passed an array as the first parameter! "
                "Try shap.waterfall_plot(explainer.base_values[0], values[0], X[0]) or "
                "for multi-output models try "
                "shap.waterfall_plot(explainer.base_values[0], values[0][0], X[0])."
            )

        if isinstance(features, pd.core.series.Series):
            if feature_names is None:
                feature_names = list(features.index)
            features = features.values

        # fallback feature names
        if feature_names is None:
            feature_names = np.array(
                [Explainer.labels["FEATURE"] % str(i) for i in range(len(values))]
            )
        # init variables we use for tracking the plot locations
        num_features = min(max_display, len(values))
        rng = range(num_features - 1, -1, -1)

        if num_features == len(values):
            num_individual = num_features
        else:
            num_individual = num_features - 1

        for i in range(num_individual):
            sval = values[order[i]]
            loc -= sval
            if sval >= 0:
                pos_inds.append(rng[i])
                pos_widths.append(sval)
                if lower_bounds is not None:
                    pos_low.append(lower_bounds[order[i]])
                    pos_high.append(upper_bounds[order[i]])
                pos_lefts.append(loc)
            else:
                neg_inds.append(rng[i])
                neg_widths.append(sval)
                if lower_bounds is not None:
                    neg_low.append(lower_bounds[order[i]])
                    neg_high.append(upper_bounds[order[i]])
                neg_lefts.append(loc)
            if num_individual != num_features or i + 4 < num_individual:
                #pl.plot([loc, loc], [rng[i] -1 - 0.4, rng[i] + 0.4], color="#bbbbbb", linestyle="--", linewidth=0.5, zorder=-1)
                x=[loc, loc]
                y=[rng[i] -1 - 0.4, rng[i] + 0.4]
                color="#bbbbbb"
                linestyle="--"
                linewidth=0.5
                zorder= -1
                output_pt = {
                    "x": x,
                    "y": y,
                    "color": color,
                    "linestyle": linestyle,
                    "linewidth": linewidth,
                    "zorder": zorder
                }
                output_pts.append(output_pt)
            if features is None:
                yticklabels[rng[i]] = feature_names[order[i]]
            else:
                yticklabels[rng[i]] = (
                    self.format_value(features[order[i]], "%0.03f")
                    + " = "
                    + feature_names[order[i]]
                )

            # add a last grouped feature to represent the impact of all the features we didn't show
            if num_features < len(shap_values):
                yticklabels[0] = "%d other features" % (
                    len(shap_values) - num_features + 1
                )
                remaining_impact = expected_value - loc
                if remaining_impact < 0:
                    pos_inds.append(0)
                    pos_widths.append(-remaining_impact)
                    pos_lefts.append(loc + remaining_impact)
                    c = colors.red_rgb
                else:
                    neg_inds.append(0)
                    neg_widths.append(-remaining_impact)
                    neg_lefts.append(loc + remaining_impact)
                    c = colors.blue_rgb

            points = (
                pos_lefts
                + list(np.array(pos_lefts) + np.array(pos_widths))
                + neg_lefts
                + list(np.array(neg_lefts) + np.array(neg_widths))
            )
            dataw = np.max(points) - np.min(points)

            # draw invisible bars just for sizing the axes
            label_padding = np.array([0.1 * dataw if w < 1 else 0 for w in pos_widths])
            p1 = pos_inds
            p2 = np.array(pos_widths).tolist() + label_padding + 0.02 * dataw
            p2 = p2.tolist()
            p3 = np.array(pos_lefts).tolist() - 0.01 * dataw
            p3 = p3.tolist()

            output_barh = {
                "pos_inds": p1,
                "pos_width": p2,
                "left": p3
            }
            jsonified = json.dumps(output_barh)
            pp = pprint.PrettyPrinter(indent=4)
            pp.pprint(jsonified)

            pl.barh(
                pos_inds,
                np.array(pos_widths) + label_padding + 0.02 * dataw,
                left=np.array(pos_lefts) - 0.01 * dataw,
                color=colors.red_rgb,
                alpha=0,
            )

    def format_value(self, s, format_str):
        """Strips trailing zeros and uses a unicode minus sign."""

        if not issubclass(type(s), str):
            s = format_str % s
        s = re.sub(r"\.?0+$", "", s)
        if s[0] == "-":
            s = u"\u2212" + s[1:]
        return s

    def lch2lab(lch):
        """CIE-LCH to CIE-LAB color space conversion.
        LCH is the cylindrical representation of the LAB (Cartesian) colorspace
        Parameters
        ----------
        lch : array_like
            The N-D image in CIE-LCH format. The last (``N+1``-th) dimension must
            have at least 3 elements, corresponding to the ``L``, ``a``, and ``b``
            color channels.  Subsequent elements are copied.
        Returns
        -------
        out : ndarray
            The image in LAB format, with same shape as input `lch`.
        Raises
        ------
        ValueError
            If `lch` does not have at least 3 color channels (i.e. l, c, h).
        Examples
        --------
        >>> from skimage import data
        >>> from skimage.color import rgb2lab, lch2lab
        >>> img = data.astronaut()
        >>> img_lab = rgb2lab(img)
        >>> img_lch = lab2lch(img_lab)
        >>> img_lab2 = lch2lab(img_lch)
        """
        lch = _prepare_lab_array(lch)

        c, h = lch[..., 1], lch[..., 2]
        lch[..., 1], lch[..., 2] = c * np.cos(h), c * np.sin(h)
        return lch

    def _prepare_lab_array(self, arr):
        """Ensure input for lab2lch, lch2lab are well-posed.
        Arrays must be in floating point and have at least 3 elements in
        last dimension.  Return a new array.
        """
        arr = np.asarray(arr)
        shape = arr.shape
        if shape[-1] < 3:
            raise ValueError("Input array has less than 3 color channels")
        return img_as_float(arr, force_copy=True)

    def img_as_float(self, image, force_copy=False):
        """Convert an image to floating point format.
        This function is similar to `img_as_float64`, but will not convert
        lower-precision floating point arrays to `float64`.
        Parameters
        ----------
        image : ndarray
            Input image.
        force_copy : bool, optional
            Force a copy of the data, irrespective of its current dtype.
        Returns
        -------
        out : ndarray of float
            Output image.
        Notes
        -----
        The range of a floating point image is [0.0, 1.0] or [-1.0, 1.0] when
        converting from unsigned or signed datatypes, respectively.
        If the input image has a float type, intensity values are not modified
        and can be outside the ranges [0.0, 1.0] or [-1.0, 1.0].
        """
        return convert(self.image, np.floating, force_copy)

