//
//  RootViewController.m
//  LLM 4 4s - Material Design 3 UI for the 4S local LLM engine.
//
//  Frames only (no Auto Layout on a fixed 320x480 screen) and every colour
//  comes from one MD3 dark tonal palette generated from a warm orange seed,
//  so the app bar, bubbles and sheets all share the same tonal family.
//
//  Deliberately avoids every post-iOS-6 API: no maskedCorners, no
//  cornerCurve, no textContainerInset, no UIAlertController, no vibrancy.
//
#import "RootViewController.h"
#import <QuartzCore/QuartzCore.h>
#import <string.h>
#import "sllm.h"

#pragma mark - palette (MD3 dark, orange seed)

#define RGBC(r, g, b, a) \
    [UIColor colorWithRed:(r) / 255.0 green:(g) / 255.0 blue:(b) / 255.0 alpha:(a)]

#define C_SURF         RGBC(0x1A, 0x11, 0x10, 1)   /* surface                 */
#define C_SURF_C       RGBC(0x27, 0x1D, 0x1B, 1)   /* surfaceContainer        */
#define C_SURF_HI      RGBC(0x32, 0x28, 0x25, 1)   /* surfaceContainerHigh    */
#define C_SURF_MAX     RGBC(0x3D, 0x32, 0x30, 1)   /* surfaceContainerHighest */
#define C_ON_SURF      RGBC(0xF1, 0xDF, 0xDA, 1)
#define C_ON_VAR       RGBC(0xCF, 0xBB, 0xB5, 1)
#define C_OUTLINE_V    RGBC(0x4A, 0x3C, 0x38, 1)
#define C_PRIMARY      RGBC(0xFF, 0xB5, 0x9B, 1)
#define C_ON_PRIMARY   RGBC(0x5B, 0x19, 0x00, 1)
#define C_PRIM_CONT    RGBC(0x7F, 0x2C, 0x00, 1)
#define C_ON_PRIM_CONT RGBC(0xFF, 0xDB, 0xCF, 1)
#define C_LOG          RGBC(0xD9, 0xC0, 0x8C, 1)

#define BAR_H     50
#define CHIP_H    28
#define CHIP_W    64
#define INPUT_H   58
#define MARGIN    14
#define BUBBLE_R  18
#define STATUS_H  38
#define INPUT_BOX_H 44

static void round_view(UIView *v, CGFloat r)
{
    v.layer.cornerRadius = r;
    v.layer.masksToBounds = YES;
}

#pragma mark - small helpers

static void push_id(int *buf, int *n, int cap, int id)
{
    if (*n < cap) buf[(*n)++] = id;
}

static void push_text(SllmTok *t, int *buf, int *n, int cap, const char *s)
{
    if (*n < cap) *n += tok_encode(t, s, buf + *n, cap - *n);
}

/* how many leading bytes of buf form complete UTF-8 sequences */
static NSUInteger utf8_safe(const unsigned char *buf, NSUInteger n)
{
    if (n == 0) return 0;
    NSUInteger i = n - 1;
    int back = 0;
    while (i > 0 && (buf[i] & 0xC0) == 0x80 && back < 3) { i--; back++; }
    unsigned char c = buf[i];
    NSUInteger need = (c & 0x80) == 0 ? 1 : ((c & 0xE0) == 0xC0 ? 2 : ((c & 0xF0) == 0xE0 ? 3 : 4));
    return (n - i >= need) ? n : i;
}

static NSString *fmt_count(long n)
{
    if (n >= 1000000) return [NSString stringWithFormat:@"%.0fM", n / 1e6];
    if (n >= 1000)    return [NSString stringWithFormat:@"%.0fK", n / 1e3];
    return [NSString stringWithFormat:@"%ld", n];
}

#pragma mark - text field

/* UITextField's own vertical centring leaves the placeholder sitting high in a
   44 pt pill (its text rect still follows the old 31 pt control height).  Doing
   the maths here makes the caret and the placeholder share one baseline. */
@interface MDField : UITextField
@end

@implementation MDField

- (CGRect)textRectForBounds:(CGRect)b
{
    CGFloat lh = self.font.lineHeight;
    return CGRectMake(16, (b.size.height - lh) / 2.0, b.size.width - 16 - 40, lh);
}

- (CGRect)editingRectForBounds:(CGRect)b
{
    return [self textRectForBounds:b];
}

- (CGRect)placeholderRectForBounds:(CGRect)b
{
    return [self textRectForBounds:b];
}

- (CGRect)clearButtonRectForBounds:(CGRect)b
{
    CGRect o = [super clearButtonRectForBounds:b];
    o.origin.x = b.size.width - o.size.width - 12;
    o.origin.y = (b.size.height - o.size.height) / 2.0;
    return o;
}

@end

#pragma mark - model descriptor

@interface EmberModel : NSObject
@property (nonatomic, copy) NSString *path;
@property (nonatomic, copy) NSString *title;
@property (nonatomic, copy) NSString *detail;
@property (nonatomic, copy) NSString *shortName;
@property (nonatomic, assign) long params;
@property (nonatomic, assign) long bytes;
@property (nonatomic, assign) int quant;
@end

@implementation EmberModel
@end

#pragma mark - vector icons

enum { MDI_SEND = 1, MDI_MENU, MDI_CLOSE, MDI_RADIO };

@interface MDIcon : UIView
@property (nonatomic, assign) NSInteger kind;
@property (nonatomic, assign) BOOL on;
@property (nonatomic, strong) UIColor *tint;
@end

@implementation MDIcon

- (void)drawRect:(CGRect)r
{
    CGContextRef g = UIGraphicsGetCurrentContext();
    UIColor *col = self.tint ? self.tint : [UIColor whiteColor];
    CGContextSetStrokeColorWithColor(g, col.CGColor);
    CGContextSetFillColorWithColor(g, col.CGColor);
    CGContextSetLineWidth(g, 2.0);
    CGContextSetLineCap(g, kCGLineCapRound);
    CGContextSetLineJoin(g, kCGLineJoinRound);

    CGFloat w = r.size.width, h = r.size.height, cx = w / 2, cy = h / 2;

    if (self.kind == MDI_SEND) {
        CGContextMoveToPoint(g, cx, cy + 6);
        CGContextAddLineToPoint(g, cx, cy - 6);
        CGContextMoveToPoint(g, cx - 5, cy - 1);
        CGContextAddLineToPoint(g, cx, cy - 6);
        CGContextAddLineToPoint(g, cx + 5, cy - 1);
        CGContextStrokePath(g);
    } else if (self.kind == MDI_MENU) {
        for (int i = 0; i < 3; i++) {
            CGFloat y = cy - 5 + i * 5;
            CGFloat len = (i == 2) ? 5 : 9;
            CGContextMoveToPoint(g, cx - 5, y);
            CGContextAddLineToPoint(g, cx - 5 + len, y);
        }
        CGContextStrokePath(g);
    } else if (self.kind == MDI_CLOSE) {
        CGContextMoveToPoint(g, cx - 5, cy - 5);
        CGContextAddLineToPoint(g, cx + 5, cy + 5);
        CGContextMoveToPoint(g, cx + 5, cy - 5);
        CGContextAddLineToPoint(g, cx - 5, cy + 5);
        CGContextStrokePath(g);
    } else if (self.kind == MDI_RADIO) {
        CGRect o = CGRectMake(cx - 8, cy - 8, 16, 16);
        if (self.on) {
            CGContextFillEllipseInRect(g, o);
            CGContextSetStrokeColorWithColor(g,
                [UIColor colorWithRed:0.10 green:0.07 blue:0.06 alpha:1.0].CGColor);
            CGContextSetLineWidth(g, 2.2);
            CGContextMoveToPoint(g, cx - 4, cy);
            CGContextAddLineToPoint(g, cx - 1, cy + 3.5);
            CGContextAddLineToPoint(g, cx + 4.5, cy - 3.5);
            CGContextStrokePath(g);
        } else {
            CGContextSetLineWidth(g, 1.6);
            CGContextStrokeEllipseInRect(g, o);
        }
    }
}

- (void)setKind:(NSInteger)k { _kind = k; [self setNeedsDisplay]; }
- (void)setOn:(BOOL)o     { _on = o;   [self setNeedsDisplay]; }
- (void)setTint:(UIColor *)t { _tint = t; [self setNeedsDisplay]; }

@end

#pragma mark - model picker sheet

@class ModelSheet;

@protocol ModelSheetDelegate <NSObject>
- (void)modelSheet:(ModelSheet *)sheet picked:(EmberModel *)m;
- (void)modelSheetCancelled:(ModelSheet *)sheet;
@end

@interface ModelSheet : UIView {
    UIView  *_panel;
    CGFloat  _panelH;
}
@property (nonatomic, strong) NSArray *models;
@property (nonatomic, assign) NSInteger selected;
@property (nonatomic, assign) id<ModelSheetDelegate> delegate;
- (id)initWithModels:(NSArray *)models selected:(NSInteger)sel;
- (void)presentIn:(UIView *)host;
- (void)dismiss;
@end

@implementation ModelSheet

- (id)initWithModels:(NSArray *)models selected:(NSInteger)sel
{
    self = [super initWithFrame:[[UIScreen mainScreen] bounds]];
    if (!self) return nil;
    _models = models;
    _selected = sel;

    self.backgroundColor = [UIColor colorWithRed:0 green:0 blue:0 alpha:0.62];

    CGFloat rowH = 62, headH = 54, footH = 52;
    _panelH = headH + rowH * models.count + footH + 10;

    _panel = [[UIView alloc] initWithFrame:CGRectMake(0, self.bounds.size.height, 320, _panelH)];
    _panel.backgroundColor = C_SURF_C;
    _panel.layer.cornerRadius = 26;
    _panel.layer.masksToBounds = YES;
    [self addSubview:_panel];

    UIView *handle = [[UIView alloc] initWithFrame:CGRectMake(140, 8, 40, 4)];
    handle.backgroundColor = C_OUTLINE_V;
    round_view(handle, 2);
    [_panel addSubview:handle];

    UILabel *title = [[UILabel alloc] initWithFrame:CGRectMake(24, 20, 240, 24)];
    title.text = @"选择模型";
    title.font = [UIFont boldSystemFontOfSize:16];
    title.textColor = C_ON_SURF;
    title.backgroundColor = [UIColor clearColor];
    [_panel addSubview:title];

    for (NSUInteger i = 0; i < models.count; i++) {
        EmberModel *m = models[i];
        CGFloat y = headH + rowH * i;
        UIView *row = [[UIView alloc] initWithFrame:CGRectMake(12, y, 296, rowH - 6)];
        if ((NSInteger)i == sel) {
            row.backgroundColor = C_SURF_HI;
            round_view(row, 16);
        }
        [_panel addSubview:row];

        MDIcon *ic = [[MDIcon alloc] initWithFrame:CGRectMake(14, (rowH - 6) / 2 - 8, 16, 16)];
        ic.kind = MDI_RADIO;
        ic.on = ((NSInteger)i == sel);
        ic.tint = C_PRIMARY;
        ic.backgroundColor = [UIColor clearColor];
        [row addSubview:ic];

        UILabel *t = [[UILabel alloc] initWithFrame:CGRectMake(42, 9, 240, 20)];
        t.text = m.title;
        t.font = [UIFont boldSystemFontOfSize:14];
        t.textColor = C_ON_SURF;
        t.backgroundColor = [UIColor clearColor];
        [row addSubview:t];

        UILabel *d = [[UILabel alloc] initWithFrame:CGRectMake(42, 29, 240, 16)];
        d.text = m.detail;
        d.font = [UIFont systemFontOfSize:10.5];
        d.textColor = C_ON_VAR;
        d.backgroundColor = [UIColor clearColor];
        [row addSubview:d];

        UIButton *hit = [UIButton buttonWithType:UIButtonTypeCustom];
        hit.frame = row.bounds;
        hit.tag = (NSInteger)i;
        [hit addTarget:self action:@selector(rowTapped:)
     forControlEvents:UIControlEventTouchUpInside];
        [row addSubview:hit];
    }

    UIView *line = [[UIView alloc] initWithFrame:CGRectMake(24, _panelH - footH - 2, 272, 0.5)];
    line.backgroundColor = C_OUTLINE_V;
    [_panel addSubview:line];

    UIButton *cancel = [UIButton buttonWithType:UIButtonTypeCustom];
    cancel.frame = CGRectMake(24, _panelH - footH + 4, 272, 40);
    [cancel setTitle:@"取消" forState:UIControlStateNormal];
    [cancel setTitleColor:C_PRIMARY forState:UIControlStateNormal];
    cancel.titleLabel.font = [UIFont boldSystemFontOfSize:14];
    [cancel addTarget:self action:@selector(cancelTapped:)
     forControlEvents:UIControlEventTouchUpInside];
    [_panel addSubview:cancel];

    return self;
}

- (void)presentIn:(UIView *)host
{
    [host addSubview:self];
    self.alpha = 0;
    CGRect end = CGRectMake(0, self.bounds.size.height - _panelH, 320, _panelH);
    [UIView animateWithDuration:0.26 animations:^{
        self.alpha = 1;
        self->_panel.frame = end;
    }];
}

- (void)dismiss
{
    [UIView animateWithDuration:0.2 animations:^{
        self.alpha = 0;
        self->_panel.frame = CGRectMake(0, self.bounds.size.height, 320, self->_panelH);
    } completion:^(BOOL done) {
        [self removeFromSuperview];
    }];
}

- (void)rowTapped:(UIButton *)b
{
    EmberModel *m = _models[b.tag];
    [self dismiss];
    [self.delegate modelSheet:self picked:m];
}

- (void)cancelTapped:(id)s
{
    [self dismiss];
    [self.delegate modelSheetCancelled:self];
}

- (void)touchesEnded:(NSSet *)touches withEvent:(UIEvent *)e
{
    UITouch *t = [touches anyObject];
    if (!CGRectContainsPoint(_panel.frame, [t locationInView:self])) {
        [self cancelTapped:nil];
    }
}

@end

#pragma mark - chat bubble

@interface MDBubble : UIView {
    UIView *_dots[3];
    BOOL    _typing;
}
@property (nonatomic, strong) UILabel *label;
@property (nonatomic, assign) BOOL isUser;
- (void)setText:(NSString *)t;
- (void)setTyping:(BOOL)on;
@end

@implementation MDBubble

- (id)initWithFrame:(CGRect)f
{
    self = [super initWithFrame:f];
    if (!self) return nil;
    self.layer.cornerRadius = BUBBLE_R;
    self.label = [[UILabel alloc] init];
    self.label.numberOfLines = 0;
    self.label.lineBreakMode = NSLineBreakByWordWrapping;
    self.label.font = [UIFont systemFontOfSize:14.5];
    self.label.backgroundColor = [UIColor clearColor];
    [self addSubview:self.label];

    for (int i = 0; i < 3; i++) {
        UIView *d = [[UIView alloc] initWithFrame:CGRectMake(0, 0, 7, 7)];
        d.backgroundColor = C_ON_SURF;
        d.layer.cornerRadius = 3.5;
        d.hidden = YES;
        [self addSubview:d];
        _dots[i] = d;
    }
    return self;
}

/* Three dots pulsing out of phase.  UIViewAnimationOptionRepeat on alpha is
   about the only looping primitive iOS 6 gives us without a display link, and
   at 0.55 s with a 0.18 s stagger it reads as a wave rather than a blink. */
- (void)setTyping:(BOOL)on
{
    _typing = on;
    if (on) {
        self.label.hidden = YES;
        CGFloat h = 34;
        CGFloat x = self.isUser ? (320 - MARGIN - 66) : MARGIN;
        self.frame = CGRectMake(x, self.frame.origin.y, 66, h);
        self.backgroundColor = C_SURF_C;
        for (int i = 0; i < 3; i++) {
            UIView *d = _dots[i];
            d.hidden = NO;
            d.frame = CGRectMake(16 + i * 14, (h - 7) / 2.0, 7, 7);
            [d.layer removeAllAnimations];
            d.alpha = 0.28;
            [UIView animateWithDuration:0.55
                                  delay:i * 0.18
                                options:UIViewAnimationOptionRepeat |
                                        UIViewAnimationOptionAutoreverse |
                                        UIViewAnimationOptionAllowUserInteraction
                             animations:^{ d.alpha = 1.0; }
                             completion:nil];
        }
    } else {
        for (int i = 0; i < 3; i++) {
            [_dots[i].layer removeAllAnimations];
            _dots[i].hidden = YES;
            _dots[i].alpha = 1.0;
        }
        self.label.hidden = NO;
    }
}

- (void)setText:(NSString *)t
{
    if (_typing) return;               /* dots own the bubble until text arrives */
    NSString *body = t.length ? t : @" ";
    self.label.text = body;
    CGSize sz = [body sizeWithFont:self.label.font
                 constrainedToSize:CGSizeMake(280 - 28, 9999)
                     lineBreakMode:NSLineBreakByWordWrapping];
    CGFloat w = MAX(44, sz.width + 28);
    CGFloat h = MAX(32, sz.height + 21);
    CGFloat x = self.isUser ? (320 - MARGIN - w) : MARGIN;
    self.frame = CGRectMake(x, self.frame.origin.y, w, h);
    self.label.frame = CGRectMake(14, 10, w - 28, h - 21);
    self.backgroundColor = self.isUser ? C_PRIM_CONT : C_SURF_C;
    self.label.textColor = self.isUser ? C_ON_PRIM_CONT : C_ON_SURF;
}

- (void)setIsUser:(BOOL)u
{
    _isUser = u;
    self.label.textColor = u ? C_ON_PRIM_CONT : C_ON_SURF;
}

@end

#pragma mark - view controller

@interface RootViewController () <UITextFieldDelegate, UIGestureRecognizerDelegate,
                                  ModelSheetDelegate> {
    SllmModel   _model;
    SllmTok     _tok;
    SllmSampler _smp;
    BOOL        _loaded;
    BOOL        _busy;
    BOOL        _modelOk;
    BOOL        _logMode;
    BOOL        _firstToken;

    int        *_conv;
    int         _convN;
    int         _convKv;
    int         _convCap;
    int         _prefillFrom;

    NSMutableData   *_pending;
    NSMutableString *_streamText;
    UILabel         *_streamLabel;
    MDBubble        *_streamBubble;
    CGFloat          _chatY;
    CGFloat          _kbHeight;
    double           _genStart;
    double           _firstTokTime;
    int              _genCount;
    int              _lastCpu;
    double           _lastTps;
    NSTimer         *_statusTimer;
    NSString        *_stateText;
    NSMutableString *_logBuf;

    NSArray         *_models;
    NSInteger        _modelIdx;
    ModelSheet      *_sheet;
}
@property (nonatomic, strong) UIView      *topBar;
@property (nonatomic, strong) UILabel     *brand;
@property (nonatomic, strong) UIButton    *chip;
@property (nonatomic, strong) UILabel     *chipLabel;
@property (nonatomic, strong) UILabel     *status;
@property (nonatomic, strong) UIButton    *logBtn;
@property (nonatomic, strong) MDIcon      *logIcon;
@property (nonatomic, strong) UIScrollView *chat;
@property (nonatomic, strong) UITextView   *logView;
@property (nonatomic, strong) UIView       *inputBar;
@property (nonatomic, strong) UITextField  *input;
@property (nonatomic, strong) UIButton     *send;
@property (nonatomic, strong) MDIcon       *sendIcon;
@end

static __unsafe_unretained RootViewController *g_vc = nil;
static void on_log(const char *line);
static void on_token(void *ud, const char *bytes, int len);

@implementation RootViewController

#pragma mark lifecycle

- (BOOL)wantsFullScreenLayout { return YES; }

- (void)loadView
{
    UIView *v = [[UIView alloc] initWithFrame:[[UIScreen mainScreen] bounds]];
    v.backgroundColor = C_SURF;
    v.autoresizingMask = UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight;
    self.view = v;
}

- (void)viewDidLoad
{
    [super viewDidLoad];
    g_vc = self;
    _pending = [[NSMutableData alloc] init];
    _streamText = [[NSMutableString alloc] init];
    _logBuf = [[NSMutableString alloc] init];
    _chatY = 10;
    _modelIdx = 0;

    [self buildUI];
    [self layoutAll];
    [self installKeyboardDismiss];

    /* Common modes so the panel keeps ticking while the log/chat is scrolling. */
    _statusTimer = [NSTimer timerWithTimeInterval:0.5 target:self
                                           selector:@selector(statusTick:)
                                           userInfo:nil repeats:YES];
    [[NSRunLoop currentRunLoop] addTimer:_statusTimer forMode:NSRunLoopCommonModes];

    [[NSNotificationCenter defaultCenter] addObserver:self
        selector:@selector(kbWillChange:) name:UIKeyboardWillShowNotification object:nil];
    [[NSNotificationCenter defaultCenter] addObserver:self
        selector:@selector(kbWillChange:) name:UIKeyboardWillHideNotification object:nil];

    [self logLine:@"===== LLM 4 4s ====="];
    [self logLine:[NSString stringWithFormat:@"dev   %@ iOS %@",
                   [[UIDevice currentDevice] model],
                   [[UIDevice currentDevice] systemVersion]]];
    [self logLine:[NSString stringWithFormat:@"scr   %.0fx%.0f @%.0fx",
                   self.view.bounds.size.width, self.view.bounds.size.height,
                   [UIScreen mainScreen].scale]];
    [self logLine:[NSString stringWithFormat:@"mem   %ldMB free",
                   sllm_free_mb()]];

    sllm_set_log(on_log);

    _models = [self discoverModels];
    [self logLine:[NSString stringWithFormat:@"files %lu model(s)",
                   (unsigned long)_models.count]];

    if (_models.count) [self loadModel:[self preferredModel]];
    else [self engineFailed:@"bundle 里找不到 .sllm 模型"];
}

/* Reload whichever model was in use last time; a cold start on the 104M costs
   several seconds of mmap, and switching by hand every launch is tedious. */
- (EmberModel *)preferredModel
{
    NSString *want = [[NSUserDefaults standardUserDefaults] stringForKey:@"model.name"];
    if (want) {
        for (EmberModel *m in _models)
            if ([[[m.path lastPathComponent] stringByDeletingPathExtension] isEqualToString:want])
                return m;
    }
    return _models[0];
}

- (void)dealloc
{
    [_statusTimer invalidate];
    _statusTimer = nil;
    [[NSNotificationCenter defaultCenter] removeObserver:self];
    sllm_set_log(NULL);
    if (g_vc == self) g_vc = nil;
}

/* Runs whenever the view's geometry settles.  This is what fixes the input
   bar starting half off-screen: during viewDidLoad the view still reports the
   raw screen size, and the window adjusts it afterwards. */
- (void)viewDidLayoutSubviews
{
    [super viewDidLayoutSubviews];
    [self layoutAll];
}

#pragma mark keyboard

- (void)installKeyboardDismiss
{
    UITapGestureRecognizer *tap =
        [[UITapGestureRecognizer alloc] initWithTarget:self
                                                action:@selector(dismissKeyboard:)];
    tap.cancelsTouchesInView = NO;      /* buttons must still get their taps */
    tap.delegate = self;
    [self.view addGestureRecognizer:tap];
}

/* let controls handle their own taps; anything else (bubbles, empty space,
   the sheet scrim) dismisses the keyboard */
- (BOOL)gestureRecognizer:(UIGestureRecognizer *)g shouldReceiveTouch:(UITouch *)t
{
    UIView *v = t.view;
    while (v) {
        if ([v isKindOfClass:[UIControl class]] ||
            [v isKindOfClass:[UITextField class]] ||
            [v isKindOfClass:[UITextView class]]) return NO;
        v = v.superview;
    }
    return YES;
}

- (void)dismissKeyboard:(id)sender
{
    [self.view endEditing:YES];
}

- (void)kbWillChange:(NSNotification *)n
{
    BOOL show = [[n name] isEqualToString:UIKeyboardWillShowNotification];
    CGRect kb = [[n.userInfo objectForKey:UIKeyboardFrameEndUserInfoKey] CGRectValue];
    _kbHeight = show ? kb.size.height : 0;

    NSTimeInterval dur = [[n.userInfo objectForKey:UIKeyboardAnimationDurationUserInfoKey]
                             doubleValue];
    [UIView beginAnimations:nil context:NULL];
    [UIView setAnimationDuration:(dur > 0 ? dur : 0.25)];
    [UIView setAnimationCurve:[[n.userInfo objectForKey:UIKeyboardAnimationCurveUserInfoKey]
                                  integerValue]];
    [self layoutAll];
    [UIView commitAnimations];
    if (show) [self scrollChatToBottom];
}

#pragma mark ui construction

- (void)buildUI
{
    self.view.backgroundColor = C_SURF;

    _topBar = [[UIView alloc] init];
    _topBar.backgroundColor = C_SURF;
    [self.view addSubview:_topBar];

    _brand = [[UILabel alloc] init];
    _brand.text = @"LLM 4 4s";
    _brand.font = [UIFont boldSystemFontOfSize:20];
    _brand.textColor = C_ON_SURF;
    _brand.backgroundColor = [UIColor clearColor];
    [_topBar addSubview:_brand];

    _logBtn = [UIButton buttonWithType:UIButtonTypeCustom];
    [_logBtn addTarget:self action:@selector(toggleLog:)
      forControlEvents:UIControlEventTouchUpInside];
    _logIcon = [[MDIcon alloc] initWithFrame:CGRectMake(8, 8, 16, 16)];
    _logIcon.kind = MDI_MENU;
    _logIcon.tint = C_ON_VAR;
    _logIcon.backgroundColor = [UIColor clearColor];
    _logIcon.userInteractionEnabled = NO;
    [_logBtn addSubview:_logIcon];
    [_topBar addSubview:_logBtn];

    _chip = [UIButton buttonWithType:UIButtonTypeCustom];
    _chip.backgroundColor = C_SURF_HI;
    round_view(_chip, CHIP_H / 2.0);
    [_chip addTarget:self action:@selector(chipTapped:)
    forControlEvents:UIControlEventTouchUpInside];
    [_topBar addSubview:_chip];

    _chipLabel = [[UILabel alloc] init];
    _chipLabel.font = [UIFont boldSystemFontOfSize:11];
    _chipLabel.textColor = C_PRIMARY;
    _chipLabel.textAlignment = NSTextAlignmentCenter;
    _chipLabel.backgroundColor = [UIColor clearColor];
    _chipLabel.userInteractionEnabled = NO;
    _chipLabel.text = @"—";
    [_chip addSubview:_chipLabel];

    _status = [[UILabel alloc] init];
    _status.font = [UIFont systemFontOfSize:14];
    _status.textColor = C_ON_VAR;
    _status.backgroundColor = [UIColor clearColor];
    _status.numberOfLines = 2;
    _status.lineBreakMode = NSLineBreakByClipping;
    [self.view addSubview:_status];

    _chat = [[UIScrollView alloc] init];
    _chat.backgroundColor = [UIColor clearColor];
    _chat.alwaysBounceVertical = YES;
    _chat.showsVerticalScrollIndicator = NO;
    [self.view addSubview:_chat];

    _logView = [[UITextView alloc] init];
    _logView.backgroundColor = C_SURF;
    _logView.textColor = C_LOG;
    /* 13 pt Courier fits 38 columns; every engine line is formatted for that. */
    _logView.font = [UIFont fontWithName:@"Courier" size:13];
    _logView.editable = NO;
    _logView.hidden = YES;
    [self.view addSubview:_logView];

    _inputBar = [[UIView alloc] init];
    _inputBar.backgroundColor = C_SURF;
    [self.view addSubview:_inputBar];

    _input = [[MDField alloc] init];
    _input.borderStyle = UITextBorderStyleNone;
    _input.backgroundColor = C_SURF_HI;
    _input.textColor = C_ON_SURF;
    _input.font = [UIFont systemFontOfSize:15];
    _input.placeholder = @"说点什么…";
    _input.returnKeyType = UIReturnKeySend;
    _input.autocorrectionType = UITextAutocorrectionTypeNo;
    _input.autocapitalizationType = UITextAutocapitalizationTypeNone;
    _input.clearButtonMode = UITextFieldViewModeWhileEditing;
    _input.delegate = self;
    round_view(_input, INPUT_BOX_H / 2.0);
    [_inputBar addSubview:_input];

    _send = [UIButton buttonWithType:UIButtonTypeCustom];
    _send.backgroundColor = C_SURF_MAX;
    round_view(_send, INPUT_BOX_H / 2.0);
    [_send addTarget:self action:@selector(sendTapped:)
    forControlEvents:UIControlEventTouchUpInside];
    _sendIcon = [[MDIcon alloc] initWithFrame:CGRectMake(10, 10, 20, 20)];
    _sendIcon.kind = MDI_SEND;
    _sendIcon.tint = C_ON_VAR;
    _sendIcon.backgroundColor = [UIColor clearColor];
    _sendIcon.userInteractionEnabled = NO;
    [_send addSubview:_sendIcon];
    [_inputBar addSubview:_send];
}

- (void)layoutAll
{
    CGRect b = self.view.bounds;
    CGFloat W = b.size.width;
    CGFloat H = b.size.height - _kbHeight;

    /* full-screen layout: reserve the status bar ourselves so it never
       covers the app bar */
    CGFloat sb = [UIApplication sharedApplication].statusBarFrame.size.height;
    if (sb <= 0) sb = 20;

    _topBar.frame = CGRectMake(0, sb, W, BAR_H);
    /* these are subviews of _topBar, so y is relative to it.  Adding sb here
       (as an earlier revision did) shifted the whole row down by the status
       bar height and made the brand collide with the status line. */
    _brand.frame  = CGRectMake(MARGIN, 0, W - 2 * MARGIN - 110, BAR_H);
    _logBtn.frame = CGRectMake(W - MARGIN - 34, (BAR_H - 34) / 2.0, 34, 34);
    _logIcon.frame = CGRectMake(9, 9, 16, 16);
    _chip.frame = CGRectMake(W - MARGIN - 34 - 8 - CHIP_W, (BAR_H - CHIP_H) / 2.0 + 1,
                             CHIP_W, CHIP_H);
    _chipLabel.frame = CGRectMake(0, 0, CHIP_W, CHIP_H);

    _status.frame = CGRectMake(MARGIN, sb + BAR_H + 1, W - 2 * MARGIN, STATUS_H);

    CGFloat chatTop = sb + BAR_H + STATUS_H + 8;
    CGFloat inputTop = H - INPUT_H;
    _chat.frame = CGRectMake(0, chatTop, W, inputTop - chatTop);
    _logView.frame = CGRectMake(0, chatTop, W, inputTop - chatTop);
    _inputBar.frame = CGRectMake(0, inputTop, W, INPUT_H);

    _input.frame = CGRectMake(MARGIN, (INPUT_H - INPUT_BOX_H) / 2.0,
                              W - 2 * MARGIN - INPUT_BOX_H - 10, INPUT_BOX_H);
    _send.frame  = CGRectMake(W - MARGIN - INPUT_BOX_H,
                              (INPUT_H - INPUT_BOX_H) / 2.0, INPUT_BOX_H, INPUT_BOX_H);
    _sendIcon.frame = CGRectMake(12, 12, 20, 20);
}

#pragma mark logging

- (void)logLine:(NSString *)line
{
    /* One event per line, always.  The console is Courier 13 on a 320 pt screen,
       so a line must stay under ~38 characters; the engine formats its own
       messages to fit and this adds a 9-character clock. */
    NSDateFormatter *df = [[NSDateFormatter alloc] init];
    [df setDateFormat:@"HH:mm:ss "];
    NSString *full = [NSString stringWithFormat:@"%@%@\n",
                      [df stringFromDate:[NSDate date]], line];

    NSMutableString *t = [[NSMutableString alloc]
        initWithString:_logView.text ? _logView.text : @""];
    [t appendString:full];
    if ([t length] > 60000) [t deleteCharactersInRange:NSMakeRange(0, [t length] - 40000)];
    _logView.text = t;
    [_logBuf appendString:full];
    [self scrollLogToBottom];
}

- (void)saveLog
{
    if ([_logBuf length] == 0) return;
    NSString *dir = [NSSearchPathForDirectoriesInDomains(NSDocumentDirectory,
                                                         NSUserDomainMask, YES) objectAtIndex:0];
    [[_logBuf copy] writeToFile:[dir stringByAppendingPathComponent:@"engine.log"]
                     atomically:YES encoding:NSUTF8StringEncoding error:NULL];
}

- (void)scrollLogToBottom
{
    CGFloat h = _logView.contentSize.height, bh = _logView.bounds.size.height;
    if (h > bh) [_logView setContentOffset:CGPointMake(0, h - bh) animated:NO];
}

static void on_log(const char *line)
{
    if (!g_vc) return;
    NSString *s = [NSString stringWithUTF8String:line];
    [g_vc performSelectorOnMainThread:@selector(logLine:)
                           withObject:(s ? s : @"(bad utf8)") waitUntilDone:NO];
}

#pragma mark model discovery / loading

- (NSArray *)discoverModels
{
    NSMutableArray *list = [NSMutableArray array];
    /* Both shipped files are int8: 4-bit weights halve the size but cost real
       accuracy on a model this small (the 104M went from answering in Chinese to
       reciting an English identity), so the device gets the exact int8 export. */
    NSArray *builtin = [NSArray arrayWithObjects:@"minimind2s_q8", @"minimind2_i8", nil];
    for (NSString *base in builtin) {
        NSString *p = [[NSBundle mainBundle] pathForResource:base ofType:@"sllm"];
        if (!p) continue;
        EmberModel *m = [self modelForPath:p];
        if (m) [list addObject:m];
    }

    NSString *docs = [NSSearchPathForDirectoriesInDomains(NSDocumentDirectory,
                                                          NSUserDomainMask, YES) objectAtIndex:0];
    NSString *dir = [docs stringByAppendingPathComponent:@"models"];
    NSFileManager *fm = [NSFileManager defaultManager];
    NSArray *files = [fm contentsOfDirectoryAtPath:dir error:NULL];
    for (NSString *f in files) {
        if (![[f pathExtension] isEqualToString:@"sllm"]) continue;
        EmberModel *m = [self modelForPath:[dir stringByAppendingPathComponent:f]];
        if (m) [list addObject:m];
    }
    return list;
}

- (EmberModel *)modelForPath:(NSString *)path
{
    SllmProbe pi;
    if (sllm_probe([path UTF8String], &pi) != 0) return nil;

    EmberModel *m = [[EmberModel alloc] init];
    m.path = path;
    m.params = pi.n_params;
    m.bytes = pi.file_size;
    m.quant = pi.quant;

    NSString *base = [[path lastPathComponent] stringByDeletingPathExtension];
    if ([base hasPrefix:@"minimind2s"])     m.title = @"MiniMind2-S";
    else if ([base hasPrefix:@"minimind2"]) m.title = @"MiniMind2";
    else                                    m.title = base;

    m.shortName = fmt_count(pi.n_params);
    NSString *qn = pi.mixed ? @"int4/int8" :
                   (pi.quant == Q_I4 ? @"int4" : (pi.quant == Q_I8 ? @"int8" : @"fp32"));
    m.detail = [NSString stringWithFormat:@"%@ · %@ 参数 · %.0f MB · ctx %d",
                qn, fmt_count(pi.n_params), pi.file_size / 1048576.0, pi.max_seq];
    return m;
}

- (void)loadModel:(EmberModel *)m
{
    NSInteger idx = (NSInteger)[_models indexOfObject:m];
    if (idx == (NSInteger)NSNotFound) idx = 0;
    _modelIdx = idx;
    _chipLabel.text = m.shortName;
    _loaded = NO;
    [self setStateText:[NSString stringWithFormat:@"载入 %@ …", m.title]];
    _send.backgroundColor = C_SURF_MAX;
    _sendIcon.tint = C_ON_VAR;
    [self performSelectorInBackground:@selector(loadModelInBackground:) withObject:m];
}

- (void)loadModelInBackground:(EmberModel *)m
{
    @autoreleasepool {
        if (_modelOk) { sllm_free(&_model); _modelOk = NO; }

        double t0 = sllm_now_ms();
        if (sllm_load(&_model, [m.path UTF8String], 2) != 0) {
            [self performSelectorOnMainThread:@selector(engineFailed:)
                                   withObject:@"模型载入失败" waitUntilDone:NO];
            return;
        }
        _modelOk = YES;

        if (!_tok.n_vocab) {
            NSString *tp = [[NSBundle mainBundle] pathForResource:@"tokenizer" ofType:@"tok"];
            if (!tp || tok_load(&_tok, [tp UTF8String]) != 0) {
                [self performSelectorOnMainThread:@selector(engineFailed:)
                                       withObject:@"分词器载入失败" waitUntilDone:NO];
                return;
            }
        }

        sampler_init(&_smp, _model.cfg.vocab_size);
        _smp.temp = 0.85f;
        _smp.top_k = 40;
        _smp.top_p = 0.90f;
        _smp.rep_pen = 1.10f;

        free(_conv);
        _convCap = _model.cfg.max_seq;
        _convN = 0;
        _convKv = 0;
        _conv = (int *)calloc((size_t)_convCap, sizeof(int));
        if (!_conv) {
            _convCap = 0;
            [self performSelectorOnMainThread:@selector(engineFailed:)
                                   withObject:@"上下文缓冲分配失败" waitUntilDone:NO];
            return;
        }
        [self buildSystemPrefix];

        NSDictionary *info = [NSDictionary dictionaryWithObjectsAndKeys:
            [NSNumber numberWithDouble:sllm_now_ms() - t0], @"ms", nil];
        [self performSelectorOnMainThread:@selector(engineReady:)
                               withObject:info waitUntilDone:NO];
    }
}

- (void)buildSystemPrefix
{
    _convN = 0;
    _convKv = 0;
    push_id(_conv, &_convN, _convCap, _tok.special[1]);
    push_text(&_tok, _conv, &_convN, _convCap, "system\n");
    push_text(&_tok, _conv, &_convN, _convCap, "You are a helpful assistant");
    push_id(_conv, &_convN, _convCap, _tok.special[2]);
    push_text(&_tok, _conv, &_convN, _convCap, "\n");
}

- (void)engineReady:(NSDictionary *)info
{
    _loaded = YES;
    _send.backgroundColor = C_PRIMARY;
    _sendIcon.tint = C_ON_PRIMARY;
    [self setStateText:nil];
    [self logLine:[NSString stringWithFormat:@"load  %.0fms ok",
                   [[info objectForKey:@"ms"] doubleValue]]];
    [self saveLog];

    EmberModel *m = (_modelIdx < (NSInteger)_models.count) ? _models[_modelIdx] : nil;
    [self resetConversation];
    if (m) [self addSystemNote:
        [NSString stringWithFormat:@"%@ · %@ 参数 · 全程离线",
         m.title, m.shortName]];
}

- (void)engineFailed:(NSString *)why
{
    _loaded = NO;
    _send.backgroundColor = C_SURF_MAX;
    _sendIcon.tint = C_ON_VAR;
    [self logLine:[NSString stringWithFormat:@"FAIL  %@", why]];
    [self setStateText:@"模型载入失败"];
    [self addSystemNote:why];
}

#pragma mark status

- (void)setStateText:(NSString *)s
{
    _stateText = [s copy];
    [self refreshStatus];
}

/* Two rows, always live: speed and CPU on top, context and memory below.
   It keeps refreshing while the model is generating, so the line never gets
   replaced by a placeholder - it is the instrument panel, not a status badge. */
- (void)refreshStatus
{
    if (!_loaded) {
        _status.text = _stateText ? _stateText : @"载入模型…";
        return;
    }
    int cpu = _lastCpu;
    long used = sllm_mem_used_mb();
    long total = sllm_mem_total_mb();

    /* While decoding, the context grows one token at a time; the buffer index
       only catches up when the turn is committed, so show the live figure. */
    int ctx = _busy ? (_convN + _genCount) : _convN;
    if (ctx > _convCap) ctx = _convCap;

    double tps = _lastTps;
    if (_busy && _genCount > 1) {
        double el = sllm_now_ms() - _firstTokTime;
        if (el > 1.0) tps = (_genCount - 1) * 1000.0 / el;
    }

    NSString *cpuStr = (cpu < 0) ? @"--" : [NSString stringWithFormat:@"%d%%", cpu];
    NSString *memStr = (total > 0 && used >= 0)
        ? [NSString stringWithFormat:@"%ld/%ldMB", used, total]
        : @"--MB";

    _status.text = [NSString stringWithFormat:
        @"%.1f tok/s · CPU %@\nctx %d/%d · %@",
        tps, cpuStr, ctx, _convCap, memStr];
}

- (void)setTps:(double)tps
{
    _lastTps = tps;
    [self refreshStatus];
}

/* 0.5 s heartbeat: the CPU figure is a delta between two samples, so it needs a
   steady cadence, and the memory number moves on its own as pages come and go.
   Without this timer the CPU field would sit at its initial 0% forever. */
- (void)statusTick:(NSTimer *)t
{
    _lastCpu = sllm_cpu_percent();
    [self refreshStatus];
}

#pragma mark chat rendering

- (void)addSystemNote:(NSString *)text
{
    UIFont *f = [UIFont systemFontOfSize:11];
    CGSize sz = [text sizeWithFont:f constrainedToSize:CGSizeMake(284, 9999)
                     lineBreakMode:NSLineBreakByWordWrapping];
    UILabel *l = [[UILabel alloc] initWithFrame:CGRectMake(18, _chatY, 284, sz.height)];
    l.text = text;
    l.font = f;
    l.textColor = C_ON_VAR;
    l.textAlignment = NSTextAlignmentCenter;
    l.backgroundColor = [UIColor clearColor];
    l.numberOfLines = 0;
    [_chat addSubview:l];
    _chatY += sz.height + 14;
    _chat.contentSize = CGSizeMake(_chat.bounds.size.width, _chatY);
}

- (MDBubble *)addBubble:(NSString *)text user:(BOOL)user
{
    MDBubble *b = [[MDBubble alloc] initWithFrame:CGRectMake(0, _chatY, 100, 32)];
    b.isUser = user;
    [b setText:text];
    [_chat addSubview:b];
    _chatY = b.frame.origin.y + b.frame.size.height + 8;
    _chat.contentSize = CGSizeMake(_chat.bounds.size.width, _chatY);
    [self scrollChatToBottom];
    return b;
}

- (void)scrollChatToBottom
{
    CGFloat h = _chat.contentSize.height, bh = _chat.bounds.size.height;
    if (h > bh) [_chat setContentOffset:CGPointMake(0, h - bh) animated:NO];
}

- (void)resizeStreamBubble
{
    if (!_streamBubble) return;
    CGFloat y = _streamBubble.frame.origin.y;
    _streamBubble.frame = CGRectMake(0, y, 100, 32);
    [_streamBubble setText:_streamText];
    _streamLabel = _streamBubble.label;
    _chatY = _streamBubble.frame.origin.y + _streamBubble.frame.size.height + 8;
    _chat.contentSize = CGSizeMake(_chat.bounds.size.width, _chatY);
    [self scrollChatToBottom];
}

- (void)resetConversation
{
    for (UIView *v in [_chat subviews]) [v removeFromSuperview];
    _chatY = 10;
    _chat.contentSize = CGSizeZero;
    _streamBubble = nil;
    _streamLabel = nil;
    if (_modelOk) {
        [self buildSystemPrefix];
        sllm_reset(&_model);
    }
    [self setTps:0];
}

#pragma mark actions

- (void)toggleLog:(id)sender
{
    _logMode = !_logMode;
    _logView.hidden = !_logMode;
    _chat.hidden = _logMode;
    _inputBar.hidden = _logMode;
    _logIcon.kind = _logMode ? MDI_CLOSE : MDI_MENU;
    if (_logMode) {
        [self.view endEditing:YES];
        [self scrollLogToBottom];
    }
    [self layoutAll];
}

- (void)chipTapped:(id)sender
{
    if (_busy || _sheet || _models.count == 0) return;
    [self.view endEditing:YES];
    _sheet = [[ModelSheet alloc] initWithModels:_models selected:_modelIdx];
    _sheet.delegate = self;
    [_sheet presentIn:self.view];
}

- (void)modelSheet:(ModelSheet *)sheet picked:(EmberModel *)m
{
    _sheet = nil;
    if ((NSInteger)[_models indexOfObject:m] == _modelIdx && _modelOk) return;
    [[NSUserDefaults standardUserDefaults] setObject:
        [[m.path lastPathComponent] stringByDeletingPathExtension] forKey:@"model.name"];
    [self loadModel:m];
}

- (void)modelSheetCancelled:(ModelSheet *)sheet
{
    _sheet = nil;
}

#pragma mark generation

- (BOOL)textFieldShouldReturn:(UITextField *)tf
{
    [self sendTapped:nil];
    return NO;
}

- (void)sendTapped:(id)sender
{
    if (!_loaded || _busy) return;
    NSString *text = [_input.text stringByTrimmingCharactersInSet:
                      [NSCharacterSet whitespaceAndNewlineCharacterSet]];
    if (!text.length) return;
    _input.text = @"";

    /* The whole turn is tokenised here, on the main thread.  _conv is then
       read-only for the worker, and _convN is already the prompt length, so the
       status line can show a context that grows token by token. */
    [self pushUserTurn:text];
    int maxNew = [self maxNewForPrompt:_convN];
    if (maxNew < 80) {
        /* Only a sliver of context left: starting over beats answering in 40
           tokens and stopping mid-sentence. */
        [self resetContext];
        [self addSystemNote:@"上下文已重置"];
        [self pushUserTurn:text];
        maxNew = [self maxNewForPrompt:_convN];
    }

    [self addBubble:text user:YES];
    [_streamText setString:@""];
    _streamBubble = [self addBubble:@"" user:NO];
    _streamLabel = _streamBubble.label;
    [_streamBubble setTyping:YES];
    _chatY = _streamBubble.frame.origin.y + _streamBubble.frame.size.height + 8;
    _chat.contentSize = CGSizeMake(_chat.bounds.size.width, _chatY);
    [self scrollChatToBottom];

    _busy = YES;
    _firstToken = YES;
    _genCount = 0;
    _send.backgroundColor = C_SURF_MAX;
    _sendIcon.tint = C_ON_VAR;
    _genStart = sllm_now_ms();
    _firstTokTime = _genStart;
    [UIApplication sharedApplication].idleTimerDisabled = YES;
    [self refreshStatus];

    NSDictionary *job = [NSDictionary dictionaryWithObjectsAndKeys:
        [NSNumber numberWithInt:maxNew], @"max", nil];
    [self performSelectorInBackground:@selector(runGenerate:) withObject:job];
}

/* 160 new tokens is about 30 s of decoding on the 4S for the 104M model, and
   the previous 96-token ceiling was what made answers stop mid-sentence even
   though the context still had room. */
- (int)maxNewForPrompt:(int)promptLen
{
    int cap = _model.cfg.dim >= 768 ? 160 : 128;
    int room = _convCap - promptLen - 4;
    if (cap > room) cap = room;
    return cap < 1 ? 1 : cap;
}

/* Drops the conversation but keeps the model: KV cache is rewound to the bare
   system prefix, so the next turn prefills from scratch. */
- (void)resetContext
{
    [self buildSystemPrefix];
    sllm_reset(&_model);
    _convKv = 0;
    [self logLine:@"note  context reset"];
}

/* Appends <|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n, rolling the
   conversation back to the system prefix if it would not fit. */
- (void)pushUserTurn:(NSString *)text
{
    int need = (int)[[text dataUsingEncoding:NSUTF8StringEncoding] length] + 24;
    if (_convN + need >= _convCap) [self resetContext];
    _prefillFrom = _convKv;
    push_id(_conv, &_convN, _convCap, _tok.special[1]);
    push_text(&_tok, _conv, &_convN, _convCap, "user\n");
    push_text(&_tok, _conv, &_convN, _convCap, [text UTF8String]);
    push_id(_conv, &_convN, _convCap, _tok.special[2]);
    push_text(&_tok, _conv, &_convN, _convCap, "\n");
    push_id(_conv, &_convN, _convCap, _tok.special[1]);
    push_text(&_tok, _conv, &_convN, _convCap, "assistant\n");
}

- (void)runGenerate:(NSDictionary *)job
{
    @autoreleasepool {
        int maxNew = [[job objectForKey:@"max"] intValue];
        int promptLen = _convN;

        int *genIds = (int *)calloc((size_t)maxNew + 1, sizeof(int));
        int n = sllm_generate(&_model, &_tok, &_smp, _conv, promptLen, _prefillFrom,
                              maxNew, on_token, (__bridge void *)self, genIds, maxNew);

        NSMutableData *ids = [NSMutableData dataWithBytes:genIds
                                                   length:(NSUInteger)(n > 0 ? n : 0) * sizeof(int)];
        free(genIds);

        double dt = sllm_now_ms() - _genStart;
        NSNumber *tps = [NSNumber numberWithDouble:(n > 0 && dt > 0) ? (n * 1000.0 / dt) : 0];
        NSDictionary *info = [NSDictionary dictionaryWithObjectsAndKeys:
            [NSNumber numberWithInt:n], @"n",
            [NSNumber numberWithInt:maxNew], @"max",
            ids, @"ids",
            tps, @"tps",
            [NSNumber numberWithInt:_model.pos], @"pos", nil];
        [self performSelectorOnMainThread:@selector(doneGenerating:)
                               withObject:info waitUntilDone:NO];
    }
}

- (void)doneGenerating:(NSDictionary *)info
{
    int n = [[info objectForKey:@"n"] intValue];
    int maxNew = [[info objectForKey:@"max"] intValue];
    NSData *ids = [info objectForKey:@"ids"];
    const int *g = (const int *)[ids bytes];

    if (_firstToken && _streamBubble) {     /* never produced a token */
        _firstToken = NO;
        [_streamBubble setTyping:NO];
    }

    /* commit the turn: the sampled ids plus the closing <|im_end|>\n that the
       model never got to emit itself */
    for (int i = 0; i < n; i++) push_id(_conv, &_convN, _convCap, g[i]);
    push_id(_conv, &_convN, _convCap, _tok.special[2]);
    push_text(&_tok, _conv, &_convN, _convCap, "\n");
    _convKv = [[info objectForKey:@"pos"] intValue];
    if (_convKv >= _convCap - 8) [self resetContext];

    [self flushPending];
    /* Hitting the cap is not an ending: say so instead of dropping the reader
       into a half-finished sentence. */
    if (_streamText.length == 0) {
        [_streamText setString:@"（无输出）"];
        [self resizeStreamBubble];
    } else if (n >= maxNew && _streamBubble) {
        [_streamText appendString:@"…"];
        [self resizeStreamBubble];
    }

    _busy = NO;
    _send.backgroundColor = _loaded ? C_PRIMARY : C_SURF_MAX;
    _sendIcon.tint = _loaded ? C_ON_PRIMARY : C_ON_VAR;
    [UIApplication sharedApplication].idleTimerDisabled = NO;
    _stateText = nil;
    [self setTps:[[info objectForKey:@"tps"] doubleValue]];
    _streamBubble = nil;
    _streamLabel = nil;
    [self logLine:[NSString stringWithFormat:@"ui    %dt ctx%d", n, _convN]];
    [self saveLog];
}

static void on_token(void *ud, const char *bytes, int len)
{
    RootViewController *vc = (__bridge RootViewController *)ud;
    NSData *d = [[NSData alloc] initWithBytes:bytes length:(NSUInteger)len];
    [vc performSelectorOnMainThread:@selector(appendChunkData:) withObject:d
                        waitUntilDone:NO];
}

- (void)appendChunkData:(NSData *)d
{
    if (_firstToken) {
        _firstToken = NO;
        _firstTokTime = sllm_now_ms();
        [_streamBubble setTyping:NO];
    }
    _genCount++;
    [_pending appendData:d];
    NSUInteger safe = utf8_safe((const unsigned char *)[_pending bytes], [_pending length]);
    if (safe == 0) return;
    NSString *s = [[NSString alloc] initWithBytes:[_pending bytes] length:safe
                                         encoding:NSUTF8StringEncoding];
    if (s) [_streamText appendString:s];
    [_pending replaceBytesInRange:NSMakeRange(0, safe) withBytes:NULL length:0];
    [self resizeStreamBubble];
}

- (void)flushPending
{
    NSUInteger n = [_pending length];
    if (!n) return;
    /* Throw away a trailing half-character: the turn ended in the middle of a
       multi-byte sequence, and half a codepoint is not text. */
    NSUInteger safe = utf8_safe((const unsigned char *)[_pending bytes], n);
    if (safe) {
        NSString *s = [[NSString alloc] initWithBytes:[_pending bytes] length:safe
                                             encoding:NSUTF8StringEncoding];
        if (s) [_streamText appendString:s];
    }
    [_pending setLength:0];
    [self resizeStreamBubble];
}

@end
