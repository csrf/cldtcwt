#ifndef AVMM_H
#define AVMM_H

#include <string>
#include <stdexcept>

extern "C" struct AVFormatContext;
extern "C" struct AVCodecContext;
extern "C" struct AVFrame;
extern "C" struct SwsContext;

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}



namespace AV {

    void registerAll();
 
    class Packet {
    public:
        Packet();
        // Construct, initialising members and giving a zero size data.

        Packet(Packet&);
        // Copy constructor.  This should take a const, but the underlying
        // library doesn't for some reason.  In case it's doing something
        // nefarious, neither will we: caveat emptor etc.

        Packet& operator= (Packet&);
        // Copy operator.  This should take a const, but the underlying
        // library doesn't for some reason.  In case it's doing something
        // nefarious, neither will we: caveat emptor etc.

        ~Packet();
        // Free the packet

        AVPacket* get();
        const AVPacket* get() const;
        // Functions to get the underlying data structures.  Please handle
        // with care!
        
        int streamIndex() const;
        // Access to the stream number

    private:
        AVPacket packet_;
        // One of the struct's from the library which can be initialised on
        // the stack, so do so.
    };


    class Frame {
    public:
        Frame();
        Frame& operator=(Frame&&);
        Frame(int width, int height, AVPixelFormat pixelFormat = PIX_FMT_RGB24);
        Frame(Frame&&);
        ~Frame();

        const AVFrame* get() const;
        AVFrame* get();

        int getWidth() const;
        int getHeight() const;
        int getLinesize() const;
        int getFormat() const;
        const uint8_t* getData() const;
        uint64_t getBestEffortTimestamp() const;
        uint64_t getPktPos() const;
        uint64_t getPTS() const;

        AVPixelFormat pixelFormat() const;
        // Pixel storage format


        void deinterlace();
        // Deinterlace the picture

        // Temporary debugging output to tmp.ppm
        void writePPM() const;

    private:
        AVFrame* frame_;
        bool classOwnsBuffer_;
    };


    class CodecContext {
    public:

        CodecContext();
        // Initialise without allocation of a resource
        
        CodecContext(const AVCodecContext*);
        // Initialise by copying from an existing AVCodecContext.

        CodecContext(CodecContext&&);
        // Move in a new resource

        CodecContext(const CodecContext&);
        // Copy to create a new resource

        CodecContext& operator=(CodecContext&&);
        // Move in a new resource, deallocating existing one

        CodecContext& operator=(const CodecContext&);
        // Copy to create a new resource, deallocating the
        // existing one
        
        ~CodecContext();
        // Deallocate resource

        AVCodecContext* get();
        const AVCodecContext* get() const;

        void flushBuffers();
        // Flush buffers associated with the CodecContext, e.g.
        // after jumping positiosn in the stream.
        
        void open(AVCodec* codec = nullptr, 
                  AVDictionary** options = nullptr);
        // Open the context using its built-in codec field.

        bool decodeVideo(Frame* frame, const Packet& packet);
        // Decode packet into frame.  Returns true if able to decode a frame.
        
        int width() const;
        int height() const;
        AVPixelFormat pixelFormat() const;
        // Access to members of the structure

    private:
        AVCodecContext* codecContext_;
    };


    class FormatContext {
    public:
        FormatContext();
        // Initialise without allocated context
        
        FormatContext(const std::string& filename,
                      AVInputFormat* fmt = nullptr,
                      AVDictionary** options = nullptr);
        // Open a resource.

        FormatContext(FormatContext&&);
        // Move an existing FormatContext in.

        FormatContext& operator= (FormatContext&&);
        // Move an existing FormatContext in, taking care
        // to deallocate any owned resources.

        ~FormatContext();
        // Destruct, deallocating any resources
        
        AVFormatContext* get();
        const AVFormatContext* get() const;
        // Access to the underlying wrapped data
        
        void findStreamInfo(AVDictionary** options = nullptr);
        // Find stream info when there are no headers, as in MPEG.
        
        int findBestStream(enum AVMediaType type,
                           int wantedStreamNb = -1,
                           int relatedStream = -1,
                           AVCodec** decoder = nullptr);
        // Find a stream of the type, and fill in the decoder with
        // a pointer to the relevant decoder pointer.
        
        CodecContext getStreamCodecContext(int stream);
        // For the stream number specified, get an unopened
        // codec context.

        void readFrame(Packet* packet);
        // Reads the next frame in the sequence

    private:
        AVFormatContext* formatContext_;
    };
    
}

extern "C" {
#include <libswscale/swscale.h>
}


namespace SWS {

    class Context {
    public:

        Context();
        // Create without initialising

        Context(Context&&);
        // Create by moving resources from a redundant object.
        
        Context(int srcW, int srcH, AVPixelFormat srcFormat,
                int dstW, int dstH, AVPixelFormat dstFormat,
                int flags = 0,
                SwsFilter* srcFilter = nullptr,
                SwsFilter* dstFilter = nullptr,
                const double* param = nullptr);
        // Create the context using sws_getCachedContext()

        Context& operator= (Context&&);
        // Move in a new object, deallocating any previously held
        // resources.

        ~Context();
        // Deallocate the context (if it exists)
        
        SwsContext* get();
        const SwsContext* get() const;
        // Access to the underlying object

        void scale(AV::Frame* dst, const AV::Frame& src);
        // Scale src into dst.

    private:
        SwsContext* context_;
    };
}


#endif

