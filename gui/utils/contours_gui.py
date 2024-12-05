from gui.popup_windows.message_boxes import ErrorMessage


def new_contour(main_window):
    if not main_window.image_displayed:
        ErrorMessage(main_window, 'Cannot create manual contour before reading input file')
        return

    main_window.tmp_lumen_x = main_window.data['lumen'][0][main_window.display.frame]  # for Ctrl+Z
    main_window.tmp_lumen_y = main_window.data['lumen'][1][main_window.display.frame]

    main_window.display.start_contour(lumen=True)
    main_window.hide_contours_box.setChecked(False)
    main_window.contours_drawn = True

def new_eem(main_window):
    if not main_window.image_displayed:
        ErrorMessage(main_window, 'Cannot create manual EEM before reading input file')
        return

    main_window.tmp_eem_x = main_window.data['eem'][0][main_window.display.frame]  # for Ctrl+Z
    main_window.tmp_eem_y = main_window.data['eem'][1][main_window.display.frame]

    main_window.display.start_contour(lumen=False)
    main_window.hide_contours_box.setChecked(False)
    main_window.contours_drawn = True

def new_measure(main_window, index: int):
    if not main_window.image_displayed:
        ErrorMessage(main_window, 'Cannot create manual measure before reading input file')
        return

    main_window.display.start_measure(index)
    main_window.hide_contours_box.setChecked(False)

def new_reference(main_window):
    if not main_window.image_displayed:
        ErrorMessage(main_window, 'Cannot create manual reference before reading input file')
        return

    main_window.display.start_reference()
