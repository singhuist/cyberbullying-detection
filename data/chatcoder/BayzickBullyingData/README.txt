This folder contains an small subset of data from crawl of myspace groups.  The data has been manually labeled for bullying content by three independent coders.  

Each input file was split into a series of windows of 10 posts each.  For example, if the group had 100 posts.  The first window was posts 1-10, the 2nd window was posts 2-11, etc.  all the way up to 91-100.  Each WINDOW was judged to determine if there was cyberbullying content anywhere within the window.  The labels are contained in separate files.  For a window to be labeled as containing cyberbullying, at least 2 out of the 3 users had to label it as cyberbullying.

See 

Jennifer Bayzick, April Kontostathis and Lynne Edwards: Detecting the Presence of Cyberbullying Using Computer Software. Poster presentation at WebSci11, June 14-17, 2011, Koblenz Germany. 

for a description of how this data was used in one small experiment.

File list:

Human Concensus.zip - contains 11 spreadsheets.  Each spreadsheet contains the labeling information for one packet
xml packet XXXX.zip - contains a set of xml files.  Each file is a window of 10 posts (unless a thread had less than 10 posts)

XML format

This XML file does not appear to have any style information associated with it. The document tree is shown below.
      

<posts> - highlevel XML wrapper
	<post> - wrapper for single post, attributes:  id
		<user> - information on user who created the post, attributes:  id
			<username>
			<sex>
			<age>
			<city>
			<province> - contains state information
			<country>
		<date> - post date in integer format			
		<body> - post text

contact akontostathis@ursinus.edu with any questions