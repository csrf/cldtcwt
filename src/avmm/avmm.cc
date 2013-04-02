#include "avmm.h"

// Work around for definition not put into stdint in C++
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}

#include <iostream>
#include <fstream>
#include <algorithm>
#include <iterator>
namespace AV {

    void registerAll()
    {
        av_register_all();
    }


    Packet::Packet()
    {
        // Initialise to a clear packet
        av_init_packet(&packet_);
        packet_.data = nullptr;
        packet_.size = 0;
    }


    Packet::Packet(Packet& src)
    {
        av_copy_packet(&packet_, &(src.packet_));
    }


    Packet& Packet::operator= (Packet& src)
    {
        av_free_packet(&packet_);

        av_copy_packet(&packet_, &(src.packet_));

        return *this;
    }


    Packet::~Packet()
    {
        av_free_packet(&packet_);
    }


    AVPacket* Packet::get()
    {
        return &packet_;
    }


    const AVPacket* Packet::get() const
    {
        return &packet_;
    }


    int Packet::streamIndex() const
    {
        return packet_.stream_index;
    }


    FormatContext::FormatContext()
        : formatContext_(nullptr)
    {}


    FormatContext::FormatContext(FormatContext&& src)
    {
        // Take the ones provided
        formatContext_ = src.formatContext_;
        
        // Restore source object to its default state
        src.formatContext_ = nullptr;
    }


    FormatContext::FormatContext(const std::string& filename,
                                 AVInputFormat* fmt,
                                 AVDictionary** options)
        : formatContext_(nullptr)
    {
        if (avformat_open_input(&formatContext_, 
                                filename.c_str(), 
                                fmt, options)
                != 0)
            throw std::runtime_error("Failed to open FormatContext");
    }



    FormatContext& FormatContext::operator=(FormatContext&& src)
    {
        // Clear any allocated resources
        if (formatContext_)
            avformat_close_input(&formatContext_);

        // Take the ones provided
        formatContext_ = src.formatContext_;
        
        // Restore source object to its default state
        src.formatContext_ = nullptr;

        return *this;
    }


    FormatContext::~FormatContext()
    {
        // Clear any allocated resources
        if (formatContext_)
            avformat_close_input(&formatContext_);
    }


    AVFormatContext* FormatContext::get()
    {
        return formatContext_;
    }


    const AVFormatContext* FormatContext::get() const
    {
        return formatContext_;
    }


   
    void FormatContext::findStreamInfo(AVDictionary** options)
    {
        int result = avformat_find_stream_info(formatContext_,
                                               options);

        // Check for an error
        if (result < 0)
            throw std::runtime_error("Failed avformat_find_stream_info");
    }


    int FormatContext::findBestStream(enum AVMediaType type,
                                      int wantedStreamNb,
                                      int relatedStream,
                                      AVCodec** decoder)
    {
        return av_find_best_stream(formatContext_, type,
                                   wantedStreamNb, relatedStream,
                                   decoder, 0);
    }



    void FormatContext::readFrame(Packet *packet)
    {
        int result = av_read_frame(formatContext_, packet->get());
    }



    CodecContext FormatContext::getStreamCodecContext(int stream)
    {
        return CodecContext(formatContext_->streams[stream]->codec);
    }



    CodecContext::CodecContext()
        : codecContext_(nullptr)
    {
    }


    CodecContext::CodecContext(CodecContext&& src)
    {
        codecContext_ = src.codecContext_;
        src.codecContext_ = nullptr;
    }


    CodecContext::CodecContext(const CodecContext& src)
    {
        // Allocate using the codec we will use in a moment (probably
        // unnecessary)
        codecContext_ = avcodec_alloc_context3(src.codecContext_->codec);

        // Copy the context.  The new one is unopened.
        avcodec_copy_context(codecContext_, src.codecContext_);
    }


    CodecContext::CodecContext(const AVCodecContext* src)
    {
        // Allocate using the codec we will use in a moment (probably
        // unnecessary)
        codecContext_ = avcodec_alloc_context3(src->codec);

        // Copy the context.  The new one is unopened.
        avcodec_copy_context(codecContext_, src);
    }



    CodecContext& CodecContext::operator=(CodecContext&& src)
    {
        // Clear up any resources we own
        if (codecContext_) {
            avcodec_close(codecContext_);
            av_free(codecContext_);
        }

        // Steal the new resources
        codecContext_ = src.codecContext_;

        // Remove them from src
        src.codecContext_ = nullptr;
            
        return *this;
    }


    CodecContext& CodecContext::operator=(const CodecContext& src)
    {
        // Clear up any resources we own
        if (codecContext_) {
            avcodec_close(codecContext_);
            av_free(codecContext_);
        }

        // Allocate using the codec we will use in a moment (probably
        // unnecessary)
        codecContext_ = avcodec_alloc_context3(src.codecContext_->codec);

        // Copy the context.  The new one is unopened.
        avcodec_copy_context(codecContext_, src.codecContext_);
           
        return *this;
    }


    CodecContext::~CodecContext()
    {
        // Clear up any resources we own
        if (codecContext_) {
            avcodec_close(codecContext_);
            av_free(codecContext_);
        }
    }


    AVCodecContext* CodecContext::get()
    {
        return codecContext_;
    }


    const AVCodecContext* CodecContext::get() const
    {
        return codecContext_;
    }




    void CodecContext::flushBuffers()
    {
        avcodec_flush_buffers(codecContext_);
    }



    void CodecContext::open(AVCodec* codec, AVDictionary** options)
    {
        if (avcodec_open2(codecContext_, codec, options) < 0)
            throw std::runtime_error("Failed to open AVCodecContext");
    }


    bool CodecContext::decodeVideo(Frame* frame, const Packet& packet)
    {
        int gotPicture;
        avcodec_decode_video2(codecContext_, frame->get(), &gotPicture,
                              packet.get());

        return gotPicture != 0;
    }


    int CodecContext::width() const
    {
        return codecContext_->width;
    }


    int CodecContext::height() const
    {
        return codecContext_->height;
    }


    AVPixelFormat CodecContext::pixelFormat() const
    {
        return codecContext_->pix_fmt;
    }



    Frame::Frame()
     : classOwnsBuffer_(false)
    {
        frame_ = avcodec_alloc_frame();
    }


    Frame::Frame(Frame&& rval)
        : frame_(rval.frame_),
          classOwnsBuffer_(rval.classOwnsBuffer_)
    {
        // Prevent it from freeing itself
        rval.frame_ = nullptr;
        rval.classOwnsBuffer_ = false;
    }


    // Move assignment operator
    Frame& Frame::operator=(Frame&& rval)
    {
        // Free own resources
        if (classOwnsBuffer_)
            avpicture_free(reinterpret_cast<AVPicture*>(frame_));

        av_free(frame_);

        // Acquire new ones
        frame_ = rval.frame_;
        classOwnsBuffer_ = rval.classOwnsBuffer_;

        // Make sure the old object doesn't try to deallocate them
        rval.frame_ = nullptr;
        rval.classOwnsBuffer_ = false;

        return *this;
    }



    Frame::Frame(int width, int height)
    {
        // Allocate a frame
        frame_ = avcodec_alloc_frame();

        // Members need filling in
        frame_->linesize[0] = width * 3;
        frame_->format = PIX_FMT_RGB24;
        frame_->width = width;
        frame_->height = height;
        
        // Attach an appropriately sized buffer
        avpicture_alloc(reinterpret_cast<AVPicture*>(frame_),
                        PIX_FMT_RGB24,
                        frame_->width,
                        frame_->height);
        // Apparently the above cast is actually safe, as the AVPicture
        // only contains a subset of AVFrame...

        // Record that we have control over the buffer
        classOwnsBuffer_ = true;

    }


    Frame::~Frame()
    {
        if (classOwnsBuffer_)
            avpicture_free(reinterpret_cast<AVPicture*>(frame_));

        av_free(frame_);
    }


    const AVFrame* Frame::get() const
    {
        return frame_;
    }



    AVFrame* Frame::get()
    {
        return frame_;
    }


    int Frame::getWidth() const
    {
        return frame_->width;
    }


    int Frame::getHeight() const
    {
        return frame_->height;
    }


    int Frame::getLinesize() const
    {
        return frame_->linesize[0];
    }


    int Frame::getFormat() const
    {
        return frame_->format;
    }


    const uint8_t* Frame::getData() const
    {
        return frame_->data[0];
    }


    uint64_t Frame::getBestEffortTimestamp() const
    {
        return av_frame_get_best_effort_timestamp(frame_);
    }


    uint64_t Frame::getPktPos() const
    {
        return av_frame_get_pkt_pos(frame_);
    }


    uint64_t Frame::getPTS() const
    {
        return frame_->pts;
    }


    AVPixelFormat Frame::pixelFormat() const
    {
        return AVPixelFormat(frame_->format);
    }


    void Frame::deinterlace()
    {
        // Apparently, the first part of an AVFrame is an AVPicture, so
        // this approach should work cleanly!
        int result = 
            avpicture_deinterlace(reinterpret_cast<AVPicture*>(frame_), 
                                  reinterpret_cast<AVPicture*>(frame_), 
                                  pixelFormat(), getWidth(), getHeight());
                                  
        if (result == -1)
            throw std::runtime_error("Failed to deinterlace");
    }


    void Frame::writePPM() const
    {
        std::ofstream out("test.ppm");

        out << "P6\n" 
            << getWidth() << " " << getHeight() << "\n"
            << "255\n";

        for (int y = 0; y < getHeight(); ++y)
            std::copy(frame_->data[0] + (y * frame_->linesize[0]),
                      frame_->data[0] + ((y+1) * frame_->linesize[0]),
                      std::ostream_iterator<uint8_t>(out));

        out << std::endl;

        
    }

}


namespace SWS {


    Context::Context()
        : context_(nullptr)
    { }

    
    Context::Context(Context&& src)
    {
        // Acquire the new ones
        context_ = src.context_;

        // Remove them from the redundant object
        src.context_ = nullptr;
    }



    Context& Context::operator=(Context&& src)
    {
        // Deallocate any resources we own
        sws_freeContext(context_);

        // Acquire the new ones
        context_ = src.context_;

        // Remove them from the redundant object
        src.context_ = nullptr;

        return *this;
    }



    Context::Context(int srcW, int srcH, AVPixelFormat srcFormat,
                     int dstW, int dstH, AVPixelFormat dstFormat,
                     int flags,
                     SwsFilter* srcFilter,
                     SwsFilter* dstFilter,
                     const double* param)
    {
        context_ = 
            sws_getCachedContext(nullptr, srcW, srcH, srcFormat,
                                          dstW, dstH, dstFormat,
                                          flags,
                                          srcFilter, dstFilter,
                                          param);
    }


    Context::~Context()
    {
        // Deallocate anything we own (safe to call with null)
        sws_freeContext(context_);
    }


    SwsContext* Context::get()
    {
        return context_;
    }


    const SwsContext* Context::get() const
    {
        return context_;
    }


    void Context::scale(AV::Frame* dst, const AV::Frame& src)
    {
        sws_scale(context_, src.get()->data,
                  src.get()->linesize, 0, src.getHeight(),
                  dst->get()->data, dst->get()->linesize);
    }

}














