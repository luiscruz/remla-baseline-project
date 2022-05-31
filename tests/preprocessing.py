
import sys
import os

sys.path.append(os.getcwd())


def test_text_prepare():

    from src.p1_preprocessing import text_prepare

    INPUT = ['SQLite/PHP read-only?\n', 'Creating Multiple textboxes dynamically\n', 'that, self or me — which one to prefer in JavaScript?\n', 'Save PHP date string into MySQL database as timestamp\n', 'How I can fill my DropDownList with Data from a XML File in my ASP.NET Application\n', '"Programmatically trigger a jQuery-UI draggable\'s ""drag"" event"\n', 'How to get the value of a method argument via reflection in Java?\n', 'Knockout maping.fromJS for observableArray from json object. Data gets lost\n', 'Facebook Connect from Localhost, doing some weird stuff\n', 'fullcalendar prev / next click\n', 'SyntaxError: Unexpected token\n', 'What is the most effective way for float and double comparison?\n', '"""gem install rails"" fails with DNS error"\n', 'Why is listShuttle component in richFaces not getting updated?\n', 'Laravel Response::download() to show images in Laravel\n', 'What is wrong with this Rspec test\n', 'Calendar display using Java Swing\n', 'python selenium import my regular firefox profile ( add-ons)\n', 'Random Number between 2 variables/values\n', 'Altering HTTP Responses in Firefox Extension\n', 'How do I start a session in a Python web application?\n', 'Align radio buttons horizontally in django forms\n', 'Count Number of Rows in a SQLite Database\n', 'Wordpress - wp_rewrite rules\n', 'Removing sheet from Excel 2005 using PHP\n', 'php Fatal error: Function name must be a string in\n', '"How to Avoid ""Used by another process"" When using File.Copy in C#"\n', 'PHP calling a class method with class name as variable string\n', 'Vector iterator not dereferencable in for loop\n', 'mySQL Search Statement with Multiple filters\n', 'PHP: Is there any Bluetooth RFCOMM library?\n', "Alternatives to Java's Scanner class for console input\n", 'PHP not displaying errors\n', 'how to check whether two matrixes are identical in OpenCV\n', 'Open all external links open in a new tab apart from a domain\n', 'Registry.GetValue always return null\n', 'Installing eventmachine on windows 8\n', 'Calculate Throughput\n', 'Want to seperate the integer part and fractional part of float number in python\n', 'How to properly manage Tomcat web apps inside Eclipse?\n', "Having trouble setting headers in a QNetworkRequest, can't understand why\n", 'Extend Request class in Laravel 5\n', 'What is the difference between matrix() and as.matrix() in r?\n', 'Install ClickOnce without running\n', 'jQuery - selected option value contains string\n', 'Error while deploying to Tomcat: tomcatManager status code:404, ReasonPhrase:Not Found\n', 'Pretty Print Distances for iOS\n', 'error: expected unqualified-id before ‘const’ on line 8\n', 'Mocking Reflection based calls\n', 'How to set Blank excel cell using POI\n', 'XCode - Code Coverage?\n', 'Creating an Interactive bar Chart out of Google Analytics Data\n',
             "math.h compilation error: expected declaration specifiers or '...' before '('\n", 'Disadvantage of object composition over class inheritance\n', 'com.mongodb.MongoTimeoutException: Timed out after 10000 ms while waiting to connect\n', 'Python Pandas : How to skip columns when reading a file?\n', "Unable to get property '1' of undefined or null reference\n", 'PyDev-Eclipse Python not configured\n', 'Python socket server receive image\n', 'What is an Event Handle?\n', 'LD_LIBRARY_PATH ignored on Android sometimes\n', 'Coin toss with JavaScript and HTML\n', 'Most efficient way to concatenate strings?\n', 'java.lang.NullPointerException when trying to read files\n', 'Assigning a lambda to an Action<T>\n', 'Split string into a list, but keeping the split pattern\n', 'Compile C code for Windows 64\n', 'Yii Condition with multiple AND/ORs - Operator Precedence\n', 'How to show SQL queries run in the Rails console?\n', 'How to change view-state value dynamically?\n', 'Best Practice For List of Polymorphic Objects in C++\n', 'Javascript: how to get values from all selected checkboxes from a table\n', 'In python, how to convert a hex ascii string to raw internal binary string?\n', 'Magento multi language rewrites SEO fix?\n', 'ul in the form not sending to the $_POST\n', 'showing alert box using php and javascript\n', 'Google Maps API v3: Looping through and adding polylines to a map\n', 'PHP remove characters after last occurrence of a character in a string\n', 'How to use Qt Creator with Visual C++ compiler on windows?\n', 'Apache spark and python lambda\n', 'Adjusting camera for visible Three.js shape\n', 'Python imaging library show() on Windows\n', 'How to open and close a website using default browser with python\n', "PHP security: 'Nonce' or 'unique form key' problem\n", 'Spring Batch: SimpleJobRepository - Example not working\n', '"Why am I getting OleDBException ""No value given for one or more required parameters?"""\n', 'Why PHP session is deleted on page reload?\n', 'Vaadin multiple browser windows/tabs\n', 'Frustrating UIWebView Delegate Crash issue\n', 'Load factor and capacity of Hash Table\n', 'How do I format a double to a string and only show decimal digits when necessary?\n', 'Round off in JAVA\n', 'Using a USB printer with C#, with and without driver or API\n', 'Unbalanced parenthesis error with Regex\n', 'Reading an Excel file with C# causes OleDbException to be thrown\n', 'I want to compare the (dt.Rows[1].ItemArray[1].ToString()) with 1,2,3,4 : but does not Work?? show me error\n', 'java.nio.charset.MalformedInputException: Input length = 1\n', 'Join two un-related tables using JPA EntityManager\n', 'Ensuring Thread-Safety On Static Methods In C#\n', 'loop focus of input fields inside a form\n']
    EXPECTED_OUTPUT = ['sqlite php readonly', 'creating multiple textboxes dynamically', 'self one prefer javascript', 'save php date string mysql database timestamp', 'fill dropdownlist data xml file aspnet application', 'programmatically trigger jqueryui draggables drag event', 'get value method argument via reflection java', 'knockout mapingfromjs observablearray json object data gets lost', 'facebook connect localhost weird stuff', 'fullcalendar prev next click', 'syntaxerror unexpected token', 'effective way float double comparison', 'gem install rails fails dns error', 'listshuttle component richfaces getting updated', 'laravel responsedownload show images laravel', 'wrong rspec test', 'calendar display using java swing', 'python selenium import regular firefox profile addons', 'random number 2 variables values', 'altering http responses firefox extension', 'start session python web application', 'align radio buttons horizontally django forms', 'count number rows sqlite database', 'wordpress wp_rewrite rules', 'removing sheet excel 2005 using php', 'php fatal error function name must string', 'avoid used another process using filecopy c#', 'php calling class method class name variable string', 'vector iterator dereferencable loop', 'mysql search statement multiple filters', 'php bluetooth rfcomm library', 'alternatives javas scanner class console input', 'php displaying errors', 'check whether two matrixes identical opencv', 'open external links open new tab apart domain', 'registrygetvalue always return null', 'installing eventmachine windows 8', 'calculate throughput', 'want seperate integer part fractional part float number python', 'properly manage tomcat web apps inside eclipse', 'trouble setting headers qnetworkrequest cant understand', 'extend request class laravel 5', 'difference matrix asmatrix r', 'install clickonce without running', 'jquery selected option value contains string', 'error deploying tomcat tomcatmanager status code404 reasonphrasenot found', 'pretty print distances ios', 'error expected unqualifiedid const line 8', 'mocking reflection based calls', 'set blank excel cell using poi', 'xcode code coverage',
                       'creating interactive bar chart google analytics data', 'mathh compilation error expected declaration specifiers', 'disadvantage object composition class inheritance', 'commongodbmongotimeoutexception timed 10000 ms waiting connect', 'python pandas skip columns reading file', 'unable get property 1 undefined null reference', 'pydeveclipse python configured', 'python socket server receive image', 'event handle', 'ld_library_path ignored android sometimes', 'coin toss javascript html', 'efficient way concatenate strings', 'javalangnullpointerexception trying read files', 'assigning lambda actiont', 'split string list keeping split pattern', 'compile c code windows 64', 'yii condition multiple ors operator precedence', 'show sql queries run rails console', 'change viewstate value dynamically', 'best practice list polymorphic objects c++', 'javascript get values selected checkboxes table', 'python convert hex ascii string raw internal binary string', 'magento multi language rewrites seo fix', 'ul form sending _post', 'showing alert box using php javascript', 'google maps api v3 looping adding polylines map', 'php remove characters last occurrence character string', 'use qt creator visual c++ compiler windows', 'apache spark python lambda', 'adjusting camera visible threejs shape', 'python imaging library show windows', 'open close website using default browser python', 'php security nonce unique form key problem', 'spring batch simplejobrepository example working', 'getting oledbexception value given one required parameters', 'php session deleted page reload', 'vaadin multiple browser windows tabs', 'frustrating uiwebview delegate crash issue', 'load factor capacity hash table', 'format double string show decimal digits necessary', 'round java', 'using usb printer c# without driver api', 'unbalanced parenthesis error regex', 'reading excel file c# causes oledbexception thrown', 'want compare dtrows 1 itemarray 1 tostring 1 2 3 4 work show error', 'javaniocharsetmalformedinputexception input length 1', 'join two unrelated tables using jpa entitymanager', 'ensuring threadsafety static methods c#', 'loop focus input fields inside form']

    for ex, ans in zip(INPUT, EXPECTED_OUTPUT):
        if text_prepare(ex) != ans:
            return False
    return True


if __name__ == '__main__':

    if test_text_prepare():
        print("Successfully passed preprocessing test.")
    else:
        print("Preprocessing test not passed.")
