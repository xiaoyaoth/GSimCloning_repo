
// TestVisual2Dlg.cpp : implementation file
//

#include "stdafx.h"
#include "TestVisual2.h"
#include "TestVisual2Dlg.h"
#include "afxdialogex.h"

//#include "SocialForce_1.h" // analyze weight with 4 * 4 room configuration

//#include "SocialForce_5.h" // analyze weight with 4 * 4 room configuration
//#include "SocialForce_2.h" // analyze weight with 4 * 4 room configuration
//#include "SocialForce_3.h" // analyze weight with 4 * 4 room configuration
//#include "SocialForce_6.h" // updated neigbor searching strategy
//#include "SocialForceGPU.h"
//#include "SocialForce_8.h"		// previous entire parameter exploration strategy with flags
#include "SocialForce_7.h"		// new strategy old scenario
//#include "SocialForce_5.h"		// two obstacle scenario
//#include "SocialForce_9.h"			// exp5 complex scenario
//#include "SocialForceGPU2.h"
//#include "SocialForceGPU.h"		// exp5 complex scenario GPU

extern "C" void runTest();

SocialForceSimApp cloneApp;

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

// CAboutDlg dialog used for App About

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

	// Dialog Data
	enum { IDD = IDD_ABOUTBOX };

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support

	// Implementation
protected:
	DECLARE_MESSAGE_MAP()
public:
	//	afx_msg void OnLButtonDblClk(UINT nFlags, CPoint point);
};

CAboutDlg::CAboutDlg() : CDialogEx(CAboutDlg::IDD)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
	//	ON_WM_LBUTTONDBLCLK()
END_MESSAGE_MAP()


// CTestVisual2Dlg dialog



CTestVisual2Dlg::CTestVisual2Dlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(CTestVisual2Dlg::IDD, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CTestVisual2Dlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CTestVisual2Dlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_WM_TIMER()
	ON_WM_ERASEBKGND()
	ON_WM_LBUTTONDBLCLK()
	ON_WM_KEYUP()
	ON_WM_MOUSEWHEEL()
	ON_WM_KEYDOWN()
	ON_WM_KEYUP()
END_MESSAGE_MAP()


// CTestVisual2Dlg message handlers

BOOL CTestVisual2Dlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// Add "About..." menu item to system menu.

	// IDM_ABOUTBOX must be in the system command range.
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// Set the icon for this dialog.  The framework does this automatically
	//  when the application's main window is not a dialog
	SetIcon(m_hIcon, TRUE);			// Set big icon
	SetIcon(m_hIcon, FALSE);		// Set small icon

	// TODO: Add extra initialization here
	pDC = this->GetDC();
	screenWidth = 640;
	screenHeight = 640;
	MoveWindow(0, 0, screenWidth + 18, screenHeight + 40, true);
	fps = 100;
	SetTimer(1, 1000 / fps, NULL);

	cloneApp.initSimClone();

	return TRUE;  // return TRUE  unless you set the focus to a control
}

void CTestVisual2Dlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// If you add a minimize button to your dialog, you will need the code below
//  to draw the icon.  For MFC applications using the document/view model,
//  this is automatically done for you by the framework.

void CTestVisual2Dlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // device context for painting

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// Center icon in client rectangle
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// Draw the icon
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

// The system calls this function to obtain the cursor to display while the user drags
//  the minimized window.
HCURSOR CTestVisual2Dlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}


void CTestVisual2Dlg::OnTimer(UINT_PTR nIDEvent)
{
	// TODO: Add your message handler code here and/or call default
	if (!isPaused)
		cloneApp.stepApp();
	myDraw();

	if (nIDEvent == 1){

	}
	CDialogEx::OnTimer(nIDEvent);
}


BOOL CTestVisual2Dlg::OnEraseBkgnd(CDC* pDC)
{
	// TODO: Add your message handler code here and/or call default

	//return CDialogEx::OnEraseBkgnd(pDC);
	return FALSE;
}

void CTestVisual2Dlg::myDraw()
{
	CClientDC dc(this);
	_memDC.CreateCompatibleDC(NULL);
	_memBitmap.CreateCompatibleBitmap(pDC, screenWidth, screenHeight);
	CBitmap *pOldBitmap = _memDC.SelectObject(&_memBitmap);
	_memDC.SetBkColor(RGB(0, 0, 0));
	_memDC.SetBkMode(TRANSPARENT);
	CRect rect0(0, 0, screenWidth, screenHeight + 50);
	_memDC.Rectangle(rect0);

	// before draw
	_memDC.SelectStockObject(NULL_BRUSH);
	int paintId = cloneApp.paintId;
	SocialForceClone *c = cloneApp.cAll[paintId];

	// draw title
	WCHAR title[128];
	swprintf_s(title, 128, L"parent: %d, clone: %d, numElem: %d, step: %d", c->parentCloneid, c->cloneid, c->numElem, cloneApp.stepCount);
	this->SetWindowText((LPCTSTR)title);

#ifdef USE_GPU
	cloneApp.getLocAndColorFromDevice();
#endif

	// draw passive clone area 
	for (int i = 0; i < NUM_CELL; i++) {
		for (int j = 0; j < NUM_CELL; j++) {
			if (cloneApp.cAll[paintId]->takenMap[i * NUM_CELL + j]) {
				CPen p(PS_SOLID, 0, RGB(240, 240, 240));
				CBrush b(RGB(240, 240, 240));
				_memDC.SelectObject(p);
				_memDC.SelectObject(b);
				int wscale = screenWidth / NUM_CELL;
				int hscale = screenHeight / NUM_CELL;
				_memDC.Rectangle(i * wscale, j * hscale, (i + 1) * wscale, (j + 1) * hscale);
			}
		}
	}

	// draw grid
	int numLine = NUM_CELL;
	for (int i = 0; i < numLine; i++) {
		char rgb = 100;
		CPen p(PS_DOT, 0, RGB(rgb, rgb, rgb));
		_memDC.SelectObject(p);
		_memDC.MoveTo(i * screenWidth / NUM_CELL, 0);
		_memDC.LineTo(i * screenWidth / NUM_CELL, screenHeight);
		_memDC.MoveTo(0, i * screenHeight / NUM_CELL);
		_memDC.LineTo(screenWidth, i * screenHeight / NUM_CELL);
	}


	// draw wall
	for (int i = 0; i < NUM_WALLS; i++) {
		CPen p(PS_SOLID, 5, RGB(0, 0, 0));
		_memDC.SelectObject(p);
		double x = c->walls[i].sx / ENV_DIM * screenWidth;
		double y = c->walls[i].sy / ENV_DIM * screenHeight;
		_memDC.MoveTo(x, y);
		x = c->walls[i].ex / ENV_DIM * screenWidth;
		y = c->walls[i].ey / ENV_DIM * screenHeight;
		_memDC.LineTo(x, y);
	}

#ifdef DRAW_OBSTACLE
	// draw gate 
	for (int i = 0; i < NUM_PARAM; i++) {
		CPen p(PS_SOLID, 5, RGB(0, 0, 0));
		_memDC.SelectObject(p);
		double x = c->gates[i].sx / ENV_DIM * screenWidth;
		double y = c->gates[i].sy / ENV_DIM * screenHeight;
		_memDC.MoveTo(x, y);
		x = c->gates[i].ex / ENV_DIM * screenWidth;
		y = c->gates[i].ey / ENV_DIM * screenHeight;
		_memDC.LineTo(x, y);
	}
#endif
	
	// draw agent
	for (int i = 0; i < NUM_CAP; i++) {
#ifdef USE_GPU
		double2 &loc = cloneApp.debugLocHost[i];
		uchar4 &color = cloneApp.debugColorHost[i];
#else
		SocialForceAgent &ag = *c->context[i];
		double2& loc = ag.data.loc;
		uchar4& color = ag.color;
#endif
		CPen p(PS_SOLID, 2, RGB(color.x, color.y, color.z));
		CBrush b(RGB(color.x, color.y, color.z));
		_memDC.SelectObject(p);
		_memDC.SelectObject(b);
		double x = loc.x / ENV_DIM * screenWidth;
		double y = loc.y / ENV_DIM * screenHeight;
		_memDC.Ellipse(x - 3, y - 3, x + 3, y + 3);

		// draw agent id;
//		CFont font;
//		font.CreatePointFont(80, L"Consolas", &_memDC);
//		CGdiObject *pOldFont = _memDC.SelectObject(&font);
//		CPen p2(PS_SOLID, 1, RGB(0, 0, 0));
//		_memDC.SelectObject(p2);
//		CRect rect(x - 10, y - 10, x + 10, y + 10);
//		CString str;
//#ifdef USE_GPU
//		str.Format(L"%d", cloneApp.debugContextIdHost[i]);
//#else
//		str.Format(L"%d", ag.contextId);
//#endif
//		_memDC.DrawText(str, rect, DT_CENTER);
	}
	//_memDC.SelectObject(pOldFont);

	// draw test
	/*
	CPoint points[] = { CPoint(20, 12), CPoint(88, 246),
	CPoint(364, 192), CPoint(250, 48),
	CPoint(175, 38), CPoint(388, 192), CPoint(145, 125) };
	_memDC.PolyBezier(points, 7);
	*/

	// after draw
	pDC->BitBlt(0, 0, screenWidth, screenHeight, &_memDC, 0, 0, SRCCOPY);
	_memBitmap.DeleteObject();
	_memDC.DeleteDC();

	UpdateData(false);
}


void CTestVisual2Dlg::OnLButtonDblClk(UINT nFlags, CPoint point)
{
	// TODO: Add your message handler code here and/or call default
	cloneApp.paintId = (cloneApp.paintId + 1) % cloneApp.totalClone;
	CDialogEx::OnLButtonDblClk(nFlags, point);
}

BOOL CTestVisual2Dlg::OnMouseWheel(UINT nFlags, short zDelta, CPoint pt)
{
	// TODO: Add your message handler code here and/or call default
	cloneApp.paintId += zDelta / 60;
	while (cloneApp.paintId < 0) cloneApp.paintId += cloneApp.totalClone;
	cloneApp.paintId = cloneApp.paintId % cloneApp.totalClone;

	return CDialogEx::OnMouseWheel(nFlags, zDelta, pt);
}


void CTestVisual2Dlg::OnKeyDown(UINT nChar, UINT nRepCnt, UINT nFlags)
{
	// TODO: Add your message handler code here and/or call default

	CDialogEx::OnKeyDown(nChar, nRepCnt, nFlags);
}


void CTestVisual2Dlg::OnKeyUp(UINT nChar, UINT nRepCnt, UINT nFlags)
{
	// TODO: Add your message handler code here and/or call default

	if (nChar == VK_UP)
		cloneApp.paintId = (cloneApp.paintId + 1) % cloneApp.totalClone;
	if (nChar == VK_DOWN)
		cloneApp.paintId = (cloneApp.paintId - 1 + cloneApp.totalClone) % cloneApp.totalClone;
	if (nChar == VK_SPACE)
		isPaused = (isPaused + 1) % 2;

	CDialogEx::OnKeyUp(nChar, nRepCnt, nFlags);
}
