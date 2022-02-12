use std::error::Error;

use image::imageops::flip_horizontal_in_place;
use image::ImageBuffer;
use image::Rgb;
use imageproc::drawing::draw_filled_rect_mut;
use imageproc::rect::Rect;
use nokhwa::Camera;
use show_image::create_window;
use show_image::glam::UVec2;
use show_image::BoxImage;
use show_image::ImageInfo;
use show_image::PixelFormat;
use tensorflow::Graph;
use tensorflow::ImportGraphDefOptions;
use tensorflow::Session;
use tensorflow::SessionOptions;
use tensorflow::SessionRunArgs;
use tensorflow::Tensor;

#[show_image::main]
fn main() -> Result<(), Box<dyn Error>> {
    let line_colour = Rgb([0u8, 255, 0]);

    let graph = loadmodel();
    let session = Session::new(&SessionOptions::new(), &graph)?;

    // set up the Camera
    let mut camera = Camera::new(
        0, None, // Some(CameraFormat::new_from(640, 480, FrameFormat::MJPEG, 30)),
    )?;

    let mut tf_input: Vec<f32> = Default::default();

    // open stream
    camera.open_stream().unwrap();

    let window = create_window("image", Default::default())?;
    loop {
        let frame = camera.frame().unwrap();
        // println!("{}, {}", frame.width(), frame.height());

        tf_input.clear();
        for pixel in frame.pixels() {
            tf_input.push(pixel.0[2] as f32);
            tf_input.push(pixel.0[1] as f32);
            tf_input.push(pixel.0[0] as f32);
        }

        let input = Tensor::new(&[frame.height() as u64, frame.width() as u64, 3])
            .with_values(&tf_input)?;
        let min_size = Tensor::new(&[]).with_values(&[40f32])?;
        let thresholds = Tensor::new(&[3]).with_values(&[0.6f32, 0.7f32, 0.7f32])?;
        let factor = Tensor::new(&[]).with_values(&[0.709f32])?;
        let mut args = SessionRunArgs::new();
        //Load our parameters for the model
        args.add_feed(&graph.operation_by_name_required("min_size")?, 0, &min_size);
        args.add_feed(
            &graph.operation_by_name_required("thresholds")?,
            0,
            &thresholds,
        );
        args.add_feed(&graph.operation_by_name_required("factor")?, 0, &factor);

        // Load our input image
        args.add_feed(&graph.operation_by_name_required("input")?, 0, &input);

        let bbox = args.request_fetch(&graph.operation_by_name_required("box")?, 0);
        let prob = args.request_fetch(&graph.operation_by_name_required("prob")?, 0);

        session.run(&mut args)?;

        let bbox_res: Tensor<f32> = args.fetch(bbox)?;
        let prob_res: Tensor<f32> = args.fetch(prob)?;

        let mut frame = frame;

        for bbox in iter_bboxen(&bbox_res, &prob_res) {
            dbg!(bbox);
            draw_filled_rect_mut(
                &mut frame,
                Rect::at(bbox.x1 as i32, bbox.y1 as i32)
                    .of_size((bbox.x2 - bbox.x1) as u32, (bbox.y2 - bbox.y1) as u32),
                line_colour,
            );
        }

        window.set_image("image-001", as_show_image(frame))?;
    }
}

fn loadmodel() -> Graph {
    //First, we load up the graph as a byte array
    let model = include_bytes!("../mtcnn.pb");

    //Then we create a tensorflow graph from the model
    let mut graph = Graph::new();
    graph
        .import_graph_def(&*model, &ImportGraphDefOptions::new())
        .unwrap();

    graph
}

fn iter_bboxen<'a>(
    bbox_res: &'a Tensor<f32>,
    prob_res: &'a Tensor<f32>,
) -> impl Iterator<Item = BBox> + 'a {
    assert_eq!(prob_res.len() * 4, bbox_res.len());
    bbox_res
        .chunks_exact(4)
        .zip(prob_res.iter())
        .map(|(bbox, &prob)| BBox {
            y1: bbox[0],
            x1: bbox[1],
            y2: bbox[2],
            x2: bbox[3],
            prob,
        })
}

#[derive(Copy, Clone, Debug)]
pub struct BBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub prob: f32,
}

fn as_show_image(inp: ImageBuffer<Rgb<u8>, Vec<u8>>) -> BoxImage {
    let imageinfo = ImageInfo {
        pixel_format: PixelFormat::Rgb8,
        size: UVec2::new(inp.height(), inp.width()),
        stride: UVec2::new(3, inp.width() * 3),
    };
    let mut inp = inp;
    flip_horizontal_in_place(&mut inp);
    BoxImage::new(imageinfo, inp.into_raw().into_boxed_slice())
}
