// This filter implements a video decoder using the ffmpeg library.

#ifndef FFMPEG_AUDIO_VIDEO_DECODER_H_
#define FFMPEG_AUDIO_VIDEO_DECODER_H_

#include <memory>
#include <string>
#include <vector>

//#include "base/type.h"
#include "opencv2/opencv.hpp"

extern "C" {
#include "libavcodec/avcodec.h"
#include "libavdevice/avdevice.h"
#include "libavformat/avformat.h"
#include "libswscale/swscale.h"
#include "libavutil/mathematics.h"
#include "libavutil/avutil.h"
}

using std::string;

typedef int8_t int8;
typedef int16_t int16;
typedef int32_t int32;
typedef int64_t int64;
typedef uint8_t uint8;
typedef uint16_t uint16;
typedef uint32_t uint32;
typedef uint64_t uint64;

// Avoid including avcodec.h in header
struct AVFrame;
struct AVPicture;
struct AVFormatContext;
struct AVCodecContext;
struct AVCodec;
struct AVPacket;


    ///////////////////////////////////////////////////////////////////////////////
    // Generic standalone video and audio decoder, wrapping libavcodec
    // FilterDecoderFFMPEG will delegate them the actual decoding job
    class FFMPEGDecoder {
    public:

        // Should be called only once.
        static void InitFFPMEG();

        FFMPEGDecoder();
        ~FFMPEGDecoder();
        
        // Returns true is file is finished decoding.
        bool finished() const { return finished_; }
        
        // Returns the number of decoded frames.
        int frame_num() const { return frame_num_; }
        
        // Don't know if this really can fail, but users better check the return
        // value to be sure. Should not be called during decoding. Unit is seconds.
        bool Seek(double seek_time);
        
    protected:
        // Deallocate everything.  After this call, the decoder is ready
        // for another file.  Also called when anything goes wrong during
        // initialization.
        void MainReset();
        
        // Opens the nth stream available in file 'filename'.
        // Returns false in case of error. 'nth' count is 1-based.
        // If codec_name is not empty, use it to find the codec for decoding.
        // Otherwise let ffmpeg determine the codec.
        bool MainOpen(const char* filename, int nth, int type,
                      const string &codec_name);
        
        bool AllocPacket();  // Allocate AVPacket structure. Returns true if ok.
        void ClearPacket();  // Clear packet content (but not the AVPacket itself).
        
        AVFormatContext *avformat_ctx_;
        AVCodecContext  *avcodec_ctx_;
        AVCodec         *avcodec_;
        AVPacket        *packet_;       // Holds fresh data.
        uint8           *packet_data_;  // Points to unconsumed data within packet_ .
        int              packet_size_;  // Size of unconsumed data.
        
        int              id_;   // Decoded stream index.
        int              frame_num_;
        bool             finished_;
        
        // Scratch buffer for conversion.
        std::unique_ptr<uint8[]> buffer_;
        int                 buffer_size_;
        AVFrame            *tmp_frame_;
    };
    
    ///////////////////////////////////////////////////////////////////////////////
    
    class FFMPEGVideoDecoder : public FFMPEGDecoder {
    public:
        FFMPEGVideoDecoder();
        ~FFMPEGVideoDecoder();
        
        // Opens the nth video stream available in file 'filename'.
        // Returns false in case of error. note: 'nth' count is 1-based.
        // If codec_name is not empty, use that name to find the codec for decoding.
        bool Open(const char* filename, int nth, const string &codec_name);
        
        // Opens the first video stream available in file 'filename'.
        // Returns false in case of error.
        bool Open(const char* filename) {
            return Open(filename, 1, "");
        }
        
        // Decode the next frame.  Returns true if the frame is ready, false in
        // case of error or EOF.  Overwrites previous frame data.  Will block
        // if repeat_frame_ is non-zero.
        bool DecodeLoop();

		bool SeekTargetVideoFrame(int target_frame);
		bool SeekToPreKeyFrame(int target_frame);
        
        // Returns the number of consumed frames.
        int ConsumedFrames() const { return frame_num_ - repeat_frame_; }
        
        // Returns true if data is ready for GetYUVData() or GetRGBData() .
        bool have_frame() const { return (repeat_frame_ > 0); }
        
        // Returns true if current (possibly repeated) frame is a keyframe.
        bool is_key_frame() const { return is_key_frame_; }
        
        // Should be called when a decoded frame is finished being processed
        // to unblock DecodeLoop() when repeat_frame_ is 0.
        void MarkFrameConsumed() { repeat_frame_--; }
        
        // Deallocate everything. After this call the decoder is ready for
        // another file.
        void Reset();
        
        // Returns the frame's width if available, 0 otherwise.
        int width() const { return width_; }
        
        // Returns the frame's height if available, 0 otherwise.
        int height() const { return height_; }

		int frame_total() const { return frame_total_; }
        
        // Returns frame per seconds if available, 0 otherwise.
        double frame_rate() const { return frame_rate_; }
        
        // Sets frame rate.
        void set_frame_rate(double frame_rate) { frame_rate_ = frame_rate; }
        
        // Sets timebase for timestamps. Timestamps returned via timestamp() function
        // are expressed w.r.t. this timebase which specifies ticks per second.
        void set_timebase(int timebase) { timebase_ = timebase; }
        
        // Enables the use of the original timestamp if force_frame_rate is true,
        // otherwise has no effect.
        void set_keep_original_timestamp(bool flag) {
            keep_original_timestamp_ = flag;
        }
        
        // If set, first timestamp will be offset to be zero, subsequent timestamps
        // will be offset by the same amount. This also offsets the original timestamp
        // if requested via set_keep_original_timestamp(true).
        void set_timestamp_offset_to_zero(bool flag) {
            timestamp_offset_to_zero_ = flag;
        }
        
        // If true, activates framerate forcing.
        void set_force_frame_rate(bool force) { force_frame_rate_ = force; }
        
        // Returns true if framerate forcing is enabled.
        bool force_frame_rate() const { return force_frame_rate_; }
        
        // Fills pointer with decoded data as RGB (packed 24bits)
        // rgb should point to height() * ystride memory size at least
        // stride should be greater or equal to 3*width()
        // bpp = 3 for RGB, 4 for RGBA
        bool GetRGBData(uint8 *rgb, int stride, int bpp) {return true;}
        
        // Returns the timestamp of the last decoded frame
        // w.r.t. timebase_ (ticks per second) or AV_NOPTS_VALUE if unavailable.
        int64 timestamp() const { return timestamp_; }
        
        //convert avframe to opencv mat
        cv::Mat convert_avframe_to_mat();
        
    private:        
        AVFrame *yuv_frame_;
        int      width_, height_;
        int      timebase_;
		int		 frame_total_;
		AVRational frame_rate_rational_;
		AVRational time_base_rational_;
		int64_t video_first_dts_;

        bool     keep_original_timestamp_;
        bool     timestamp_offset_to_zero_;
        double   frame_rate_;
        int      repeat_frame_;
        bool     force_frame_rate_;
        bool     is_key_frame_;
        // Offset w.r.t. timebase_, expressed as fractional double.
        // Used if force_frame_rate_ is set.
        double   time_offset_in_timebase_;
        // Indicates if corresponding offset of first frame has been retrieved.
        bool     time_offset_in_timebase_set_;
        // Offset timestamp as indicated by first packet's dts. Stored w.r.t. encoded
        // videos timebase (different from timebase_ in general). If set to
        // AV_NOPTS_VALUE, no value has been recorded yet.
        int64    time_offset_dts_;
        int64    timestamp_;
    };
    
    class FFMPEGAudioDecoder : public FFMPEGDecoder {
    public:
        FFMPEGAudioDecoder();
        ~FFMPEGAudioDecoder();
        
        // Opens the nth audio stream available in file 'filename'.
        // Returns false in case of error. note: 'nth' count is 1-based.
        // If codec_name is not empty, use that name to find the codec for decoding.
        bool Open(const char* filename, int nth, double audio_frame_length,
                  const string &codec_name);
        
        // Opens the first audio stream available in file 'filename'.
        // Returns false in case of error.
        bool Open(const char* filename, double audio_frame_length) {
            return Open(filename, 1, audio_frame_length, "");
        }
        
        // Decode next frame. Returns true if data is available
        // frame_size is the minimal number of bytes requested to form a frame
        // fewer bytes can actually be returned if stream is truncated at end
        bool DecodeAudioFrame(int32 frame_size);
        
        // Deallocate everything.
        void Reset();
        
        // Returns the number of channels.
        int channels() const { return channels_; }
        
        // Returns audio sample rate (samples per second).
        int32 sample_rate() const { return sample_rate_; }
        
        // Returns (virtual) audio pos, in sample units.
        int64 sample_position() const { return sample_position_; }
        
        // Return size (in byte) of next audio frame, according to frame_num_ .
        int32 GetNextFrameSize() const;
        
        // Returns size of available data, in bytes.
        int32 DataSize() const { return data_size_; }
        
        // Fills a frame with S16_LE samples. Returns the number of bytes filled:
        // should be data_size, or 0 in case of error.
        // Data is irreversibly consumed afterward.
        int32 ConsumeFrame(float *out, const int32 data_size);
        
    private:
        // Return maximum possible frame size, in bytes.
        int32 GetMaxAudioSize() const;
        
        // Audio frame.
        std::unique_ptr<int16[]> audio_buf_;
        
        // size of one audio frame, in samples  (= 1 second of sound currently)
        // stored as float to avoid drift. Use GetNextFrameSize().
        float audio_size_;
        
        // Data available in audio_buf_, in bytes.
        int32 data_size_;
        
        // The current audio position, in sample units.
        int64 sample_position_;
        
        // Number of audio channels.
        int channels_;
        
        // Audio sample rate in samples per second.
        int32 sample_rate_;
        
        // Aligned buffer to read the decoded frame.
        int aligned_buffer_size_;
        std::unique_ptr<int16_t, void (*)(void *)> aligned_buffer_;
    };
    
#endif  // FFMPEG_AUDIO_VIDEO_DECODER_H_

