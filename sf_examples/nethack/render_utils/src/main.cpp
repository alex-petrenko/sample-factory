#include <numeric>

#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

template <typename T> void check_array_types(py::handle h, int min_dim) {
  if (h.is_none())
    throw std::invalid_argument("Array is None!");
  ;
  if (!py::isinstance<py::array>(h))
    throw std::invalid_argument("Numpy array required.");

  py::array array = py::array::ensure(h);
  if (!array.dtype().is(py::dtype::of<T>()))
    throw std::invalid_argument("Buffer dtype mismatch.");

  if (!(array.flags() & py::array::c_style))
    throw std::invalid_argument("Array isn't C contiguous.");

  py::buffer_info buf = array.request();
  if (buf.ndim < min_dim)
    throw std::invalid_argument("Wrong ndim in array");
}

void check_array_shapes(py::handle tty_chars, py::handle tty_colors,
                        py::handle tty_cursor, py::handle glyph_images,
                        py::handle out, int crop_size) {

  const auto &chars_shape = py::array::ensure(tty_chars).request().shape;
  const auto &colors_shape = py::array::ensure(tty_colors).request().shape;
  if (!std::equal(chars_shape.begin(), chars_shape.end(), colors_shape.begin()))
    throw std::invalid_argument("Shape mismatch (tty_chars, tty_colors).");

  const auto &img_shape = py::array::ensure(glyph_images).request().shape;
  if (!(img_shape[0] == 256 && img_shape[1] == 16))
    throw std::invalid_argument("Shape of glyph_images must start (256, 16)");

  int crop_height =
      (crop_size > 0) ? crop_size : chars_shape[chars_shape.size() - 2];
  int crop_width =
      (crop_size > 0) ? crop_size : chars_shape[chars_shape.size() - 1];
  const auto &out_shape = py::array::ensure(out).request().shape;
  size_t dim = out_shape.size();
  if (!(out_shape[dim - 3] == img_shape[2] &&
        out_shape[dim - 2] == (crop_height * img_shape[3]) &&
        out_shape[dim - 1] == (crop_width * img_shape[4])))
    throw std::invalid_argument("Shape mismatch (glyph_images, out).");

  const auto &cursor_shape = py::array::ensure(tty_cursor).request().shape;
  if (!(cursor_shape[cursor_shape.size() - 1] == 2))
    throw std::invalid_argument("Shape of glyph_images must be (2)");

  if (!(chars_shape.size() - 2 == cursor_shape.size() - 1 &&
        chars_shape.size() - 2 == out_shape.size() - 3)) {
    throw std::invalid_argument("Different dims for batch conversion");
  }

  for (int i = 0; i < chars_shape.size() - 2; ++i) {
    if (!(chars_shape[i] == cursor_shape[i] &&
          chars_shape[i] == out_shape[i])) {
      throw std::invalid_argument("Different batch sizes for batch conversion");
    }
  }
}

void tile_crop(py::array_t<uint8_t> tty_chars, py::array_t<int8_t> tty_colors,
               py::array_t<uint8_t> tty_cursor, py::array_t<uint8_t> images,
               py::array_t<uint8_t> out_array, int crop_size) {

  py::buffer_info chars_buff = tty_chars.request();
  py::buffer_info colors_buff = tty_colors.request();
  py::buffer_info cursor_buff = tty_cursor.request();
  py::buffer_info images_buff = images.request();
  py::buffer_info out_buff = out_array.request();

  const auto &chars_shape = chars_buff.shape;
  const auto &img_shape = images_buff.shape;
  const auto &out_shape = out_buff.shape;

  int lead_dims = chars_shape.size() - 2;
  int lead_elems = 1;
  for (int i = 0; i < lead_dims; ++i) {
    lead_elems *= chars_shape[i];
  }

  int rows = chars_shape[lead_dims + 0];
  int cols = chars_shape[lead_dims + 1];

  int img_colors = img_shape[1];
  int img_channels = img_shape[2];
  int img_rows = img_shape[3];
  int img_cols = img_shape[4];

  int out_chan = out_shape[lead_dims + 0];
  int out_rows = out_shape[lead_dims + 1];
  int out_cols = out_shape[lead_dims + 2];

  uint8_t *char_ptr = static_cast<uint8_t *>(chars_buff.ptr);
  int8_t *color_ptr = static_cast<int8_t *>(colors_buff.ptr);
  uint8_t *out_ptr = static_cast<uint8_t *>(out_buff.ptr);
  uint8_t *cur_ptr = static_cast<uint8_t *>(cursor_buff.ptr);
  uint8_t *img_ptr = static_cast<uint8_t *>(images_buff.ptr);

  int half_crop_size = crop_size / 2;

  // Strides
  int s_char_frame = rows * cols;
  int s_char_row = cols;

  int s_color_frame = rows * cols;
  int s_color_row = cols;

  int s_cursor_frame = 2;

  int s_img_col = img_cols;
  int s_img_row = img_rows * img_cols;
  int s_img_color = img_channels * img_rows * img_cols;
  int s_img_glyph = img_colors * img_channels * img_rows * img_cols;

  int s_out_frame = out_chan * out_rows * out_cols;
  int s_out_chan = out_rows * out_cols;
  int s_out_row = out_cols;
  {
    py::gil_scoped_release release;

    for (size_t i = 0; i < lead_elems; ++i) {
      auto chars_at = [char_ptr, s_char_row](int h, int w) {
        return *(char_ptr + h * s_char_row + w);
      };
      auto colors_at = [color_ptr, s_color_row](int h, int w) {
        return *(color_ptr + h * s_color_row + w);
      };
      auto img_at = [img_ptr, s_img_glyph, s_img_color, s_img_row,
                     s_img_col](int glyph, int color, int chan, int h, int w) {
        return *(img_ptr + glyph * s_img_glyph + color * s_img_color +
                 chan * s_img_row + h * s_img_col + w);
      };
      auto out_ptr_ = [out_ptr, s_out_chan, s_out_row](int chan, int h, int w) {
        return (out_ptr + chan * s_out_chan + h * s_out_row + w);
      };

      int start_h = (crop_size > 0) ? cur_ptr[0] - half_crop_size : 0;
      int start_w = (crop_size > 0) ? cur_ptr[1] - half_crop_size : 0;

      int max_r = (crop_size > 0) ? crop_size : rows;
      int max_c = (crop_size > 0) ? crop_size : cols;
      for (size_t r = 0; r < max_r; ++r) {
        int h = r + start_h;
        for (size_t c = 0; c < max_c; ++c) {
          int w = c + start_w;
          for (size_t i_chan = 0; i_chan < img_channels; ++i_chan) {
            for (size_t i_r = 0; i_r < img_rows; ++i_r) {
              for (size_t i_c = 0; i_c < img_cols; ++i_c) {

                if ((h < 0 || h >= rows || w < 0 || w >= cols)) {
                  *out_ptr_(i_chan, r * img_rows + i_r, c * img_cols + i_c) = 0;
                } else {
                  int this_glyph = chars_at(h, w);
                  int this_color = colors_at(h, w);
                  *out_ptr_(i_chan, r * img_rows + i_r, c * img_cols + i_c) =
                      img_at(this_glyph, this_color, i_chan, i_r, i_c);
                }
              }
            }
          }
        }
      }
      char_ptr += s_char_frame;
      color_ptr += s_color_frame;
      cur_ptr += s_cursor_frame;
      out_ptr += s_out_frame;
    }
  }
}
void render_crop(py::object tty_chars, py::object tty_colors,
                 py::object tty_cursor, py::object images, py::object out_array,
                 int crop_size) {

  check_array_types<uint8_t>(tty_chars, 2);
  check_array_types<int8_t>(tty_colors, 2);
  check_array_types<uint8_t>(tty_cursor, 1);
  check_array_types<uint8_t>(images, 5);
  check_array_types<uint8_t>(out_array, 3);
  check_array_shapes(tty_chars, tty_colors, tty_cursor, images, out_array,
                     crop_size);

  tile_crop(tty_chars, tty_colors, tty_cursor, images, out_array, crop_size);
}

namespace py = pybind11;

PYBIND11_MODULE(render_utils, m) {
  m.doc() = R"pbdoc(
        A module to turn glyphs into the screen in pixels
        -----------------------
    )pbdoc";

  m.def("render_crop", &render_crop, py::arg("tty_chars"),
        py::arg("tty_colors"), py::arg("tty_cursor"), py::arg("images"),
        py::arg("out_array"), py::arg("crop_size") = 12);

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
