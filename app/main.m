//
//  main.m / AppDelegate.m
//  LLM 4 4s - local Transformer LLM on an iPhone 4S (iOS 6.1.3)
//
#import <UIKit/UIKit.h>
#import <stdio.h>
#import "RootViewController.h"

/* Write the reason to Documents before the process dies: an ObjC exception
   in a UI callback otherwise leaves no trace at all on iOS 6. */
static void llm44s_uncaught(NSException *e)
{
    NSString *s = [NSString stringWithFormat:
        @"=== uncaught ===\n%@\n%@\n\n%@\n",
        [e name], [e reason],
        [[e callStackSymbols] componentsJoinedByString:@"\n"]];
    NSArray *dirs = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory,
                                                        NSUserDomainMask, YES);
    if ([dirs count]) {
        [s writeToFile:[[dirs objectAtIndex:0] stringByAppendingPathComponent:@"crash.log"]
            atomically:YES encoding:NSUTF8StringEncoding error:NULL];
    }
    fprintf(stderr, "%s", [s UTF8String]);
}

@interface AppDelegate : UIResponder <UIApplicationDelegate>
@property (nonatomic, strong) UIWindow *window;
@end

@implementation AppDelegate

- (BOOL)application:(UIApplication *)application
didFinishLaunchingWithOptions:(NSDictionary *)launchOptions
{
    NSSetUncaughtExceptionHandler(&llm44s_uncaught);

    self.window = [[UIWindow alloc] initWithFrame:[[UIScreen mainScreen] bounds]];
    self.window.backgroundColor = [UIColor blackColor];
    RootViewController *root = [[RootViewController alloc] initWithNibName:nil bundle:nil];
    self.window.rootViewController = root;
    [self.window makeKeyAndVisible];
    return YES;
}

- (void)applicationDidReceiveMemoryWarning:(UIApplication *)application
{
    NSLog(@"[llm44s] memory warning");
}

@end

int main(int argc, char *argv[])
{
    @autoreleasepool {
        return UIApplicationMain(argc, argv, nil, @"AppDelegate");
    }
}
